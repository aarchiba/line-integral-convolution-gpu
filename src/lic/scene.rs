//! On-screen LIC scene driven by user-provided generators.
//!
//! The compiler requires a [`LicSceneSpec`] up front: three generator
//! functions (ink, vector field, kernel) that each take the time in seconds.
//! Initialization and animation both call the *same* generators — init at
//! `t = 0`, the update systems at the current time — so a fixed input is
//! simply a generator that ignores `t`, written exactly once:
//!
//! ```ignore
//! LicSceneSpec::new(
//!     640, 360,
//!     |w, h, _t| generate_noise_grayscale(w, h, 0.06), // fixed ink
//!     |w, h, _t| analytical::vortex(w, h, (cx(w), cy(h)), 60.0), // fixed field
//!     |t| LicParams::triangular_kernel(breathing_radius(t)), // animated kernel
//! )
//! ```
//!
//! Schedule only the update systems you need: omit [`update_lic_ink`] and
//! [`update_lic_field`] for fixed inputs (no redundant per-frame work).

use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::*;
use bevy::sprite::MeshMaterial2d;

use crate::lic::material::LicMaterial;
use crate::lic::params::LicParams;

/// Ink generator: grayscale values in `[0,1]`, row-major, `w * h` entries.
pub type NoiseGenerator = Box<dyn Fn(u32, u32, f32) -> Vec<f32> + Send + Sync + 'static>;
/// Vector field generator: pixel-space interleaved `(dx, dy)`, row-major,
/// `w * h * 2` entries.
pub type FieldGenerator = Box<dyn Fn(u32, u32, f32) -> Vec<f32> + Send + Sync + 'static>;
/// Kernel generator: full [`LicParams`] for time `t`.
pub type KernelGenerator = Box<dyn Fn(f32) -> LicParams + Send + Sync + 'static>;

/// Required specification of the initial ink, vector field, and kernel.
/// Passed to `build_app`; there are no built-in defaults.
#[derive(Resource)]
pub struct LicSceneSpec {
    pub width: u32,
    pub height: u32,
    pub noise: NoiseGenerator,
    pub field: FieldGenerator,
    pub kernel: KernelGenerator,
}

impl LicSceneSpec {
    pub fn new(
        width: u32,
        height: u32,
        noise: impl Fn(u32, u32, f32) -> Vec<f32> + Send + Sync + 'static,
        field: impl Fn(u32, u32, f32) -> Vec<f32> + Send + Sync + 'static,
        kernel: impl Fn(f32) -> LicParams + Send + Sync + 'static,
    ) -> Self {
        Self {
            width,
            height,
            noise: Box::new(noise),
            field: Box::new(field),
            kernel: Box::new(kernel),
        }
    }
}

/// Handles for the live scene, for update systems to find the material.
#[derive(Resource, Debug, Clone)]
pub struct LicScene {
    pub material: Handle<LicMaterial>,
    pub width: u32,
    pub height: u32,
}

fn noise_to_rgba(gray: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(gray.len() * 4);
    for v in gray {
        let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
        bytes.extend_from_slice(&[byte, byte, byte, 255]);
    }
    bytes
}

/// Pixel-space field → little-endian `f16` pairs (`Rg16Float`).
fn field_to_f16(pixel: &[f32], width: u32, height: u32) -> Vec<u16> {
    let mut uv = Vec::with_capacity(pixel.len());
    for pair in pixel.as_chunks::<2>().0 {
        uv.push(half::f16::from_f32(pair[0] / width as f32).to_bits());
        uv.push(half::f16::from_f32(pair[1] / height as f32).to_bits());
    }
    uv
}

fn scene_image(
    images: &mut Assets<Image>,
    width: u32,
    height: u32,
    data: Vec<u8>,
    format: TextureFormat,
) -> Handle<Image> {
    images.add(Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        format,
        RenderAssetUsages::RENDER_WORLD,
    ))
}

/// Build the scene from the spec, evaluating every generator at `t = 0`.
pub fn setup_lic_scene(
    spec: Res<LicSceneSpec>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LicMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let noise_handle = scene_image(
        &mut images,
        spec.width,
        spec.height,
        noise_to_rgba(&(spec.noise)(spec.width, spec.height, 0.0)),
        TextureFormat::Rgba8UnormSrgb,
    );
    let field_handle = scene_image(
        &mut images,
        spec.width,
        spec.height,
        bytemuck::cast_slice(&field_to_f16(
            &(spec.field)(spec.width, spec.height, 0.0),
            spec.width,
            spec.height,
        ))
        .to_vec(),
        TextureFormat::Rg16Float,
    );

    let material_handle = materials.add(LicMaterial {
        noise_texture: noise_handle,
        vector_field_texture: field_handle,
        params: (spec.kernel)(0.0),
    });
    commands.insert_resource(LicScene {
        material: material_handle.clone(),
        width: spec.width,
        height: spec.height,
    });

    // Fullscreen quad (window is 1280x720; use a 16:9 spec to avoid distortion).
    let quad = Mesh::from(Rectangle::new(1280.0, 720.0));
    commands.spawn((
        Mesh2d(meshes.add(quad)),
        MeshMaterial2d(material_handle),
        Transform::default(),
    ));

    commands.spawn((Camera2d, Transform::default()));
}

/// Re-evaluate the ink generator at the current time and rewrite the texture.
pub fn update_lic_ink(
    spec: Res<LicSceneSpec>,
    scene: Res<LicScene>,
    time: Res<Time>,
    mut images: ResMut<Assets<Image>>,
    materials: Res<Assets<LicMaterial>>,
) {
    let Some(mat) = materials.get(&scene.material) else {
        return;
    };
    let Some(img) = images.get_mut(&mat.noise_texture) else {
        return;
    };
    let fresh = noise_to_rgba(&(spec.noise)(scene.width, scene.height, time.elapsed_secs()));
    debug_assert_eq!(fresh.len(), img.data.len());
    img.data.copy_from_slice(&fresh);
}

/// Re-evaluate the field generator at the current time and rewrite the texture.
pub fn update_lic_field(
    spec: Res<LicSceneSpec>,
    scene: Res<LicScene>,
    time: Res<Time>,
    mut images: ResMut<Assets<Image>>,
    materials: Res<Assets<LicMaterial>>,
) {
    let Some(mat) = materials.get(&scene.material) else {
        return;
    };
    let Some(img) = images.get_mut(&mat.vector_field_texture) else {
        return;
    };
    let fresh = field_to_f16(
        &(spec.field)(scene.width, scene.height, time.elapsed_secs()),
        scene.width,
        scene.height,
    );
    let bytes: &[u8] = bytemuck::cast_slice(&fresh);
    debug_assert_eq!(bytes.len(), img.data.len());
    img.data.copy_from_slice(bytes);
}

/// Re-evaluate the kernel generator at the current time.
pub fn update_lic_kernel(
    spec: Res<LicSceneSpec>,
    scene: Res<LicScene>,
    time: Res<Time>,
    mut materials: ResMut<Assets<LicMaterial>>,
) {
    if let Some(mat) = materials.get_mut(&scene.material) {
        mat.params = (spec.kernel)(time.elapsed_secs());
    }
}
