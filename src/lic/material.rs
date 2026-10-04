//! LIC material (ticket 5: streamline integration)

use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::*;
use bevy::sprite::{Material2d, Material2dPlugin};

use crate::lic::params::LicParams;
use crate::lic::vector_field::analytical;
use crate::noise::generate_noise_grayscale;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct LicMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub noise_texture: Handle<Image>,

    #[texture(2)]
    #[sampler(3)]
    pub vector_field_texture: Handle<Image>,

    #[uniform(4)]
    pub params: LicParams,
}

impl Material2d for LicMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/lic.wgsl".into()
    }

    fn alpha_mode(&self) -> bevy::sprite::AlphaMode2d {
        bevy::sprite::AlphaMode2d::Opaque
    }
}

pub fn lic_material_plugin(app: &mut App) {
    app.add_plugins(Material2dPlugin::<LicMaterial>::default());
}

/// Demo scene texture size (16:9, matches the window so the vortex stays circular).
pub const DEMO_WIDTH: u32 = 640;
pub const DEMO_HEIGHT: u32 = 360;
/// Vortex strength in pixel units (matches `examples/lic_demo.rs` visuals).
pub const DEMO_VORTEX_STRENGTH: f32 = 60.0;
/// Triangular kernel radius (matches `examples/lic_demo.rs`).
pub const DEMO_KERNEL_RADIUS: u32 = 12;

/// Spawn a fullscreen quad rendering live LIC output: fine-grain Perlin noise
/// smeared along a central vortex field. This is what `cargo run` displays.
pub fn setup_lic_display(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LicMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // Noise input (RGBA8). Finer grain than the old noise display so the
    // kernel span covers several noise features and streaks are visible.
    let noise = generate_noise_grayscale(DEMO_WIDTH, DEMO_HEIGHT, 0.06);
    let mut noise_bytes = Vec::with_capacity((DEMO_WIDTH * DEMO_HEIGHT * 4) as usize);
    for v in &noise {
        let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
        noise_bytes.extend_from_slice(&[byte, byte, byte, 255]);
    }
    let noise_image = Image::new(
        Extent3d {
            width: DEMO_WIDTH,
            height: DEMO_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        noise_bytes,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    let noise_handle = images.add(noise_image);

    // Vortex field converted pixel-space -> UV-space (divide by dimensions),
    // stored as RG16Float per LIC_DESIGN.md (filterable, unlike Rg32Float).
    let cx = (DEMO_WIDTH - 1) as f32 / 2.0;
    let cy = (DEMO_HEIGHT - 1) as f32 / 2.0;
    let pixel = analytical::vortex(DEMO_WIDTH, DEMO_HEIGHT, (cx, cy), DEMO_VORTEX_STRENGTH);
    let mut uv: Vec<u16> = Vec::with_capacity(pixel.len());
    for pair in pixel.chunks_exact(2) {
        uv.push(half::f16::from_f32(pair[0] / DEMO_WIDTH as f32).to_bits());
        uv.push(half::f16::from_f32(pair[1] / DEMO_HEIGHT as f32).to_bits());
    }
    let field_image = Image::new(
        Extent3d {
            width: DEMO_WIDTH,
            height: DEMO_HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        bytemuck::cast_slice(&uv).to_vec(),
        TextureFormat::Rg16Float,
        RenderAssetUsages::RENDER_WORLD,
    );
    let field_handle = images.add(field_image);

    let material = LicMaterial {
        noise_texture: noise_handle,
        vector_field_texture: field_handle,
        params: LicParams::triangular_kernel(DEMO_KERNEL_RADIUS),
    };
    let material_handle = materials.add(material);

    // Fullscreen quad (window is 1280x720, texture aspect matches).
    let quad = Mesh::from(Rectangle::new(1280.0, 720.0));
    let mesh_handle = meshes.add(quad);

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(material_handle),
        Transform::default(),
    ));

    commands.spawn((Camera2d, Transform::default()));
}