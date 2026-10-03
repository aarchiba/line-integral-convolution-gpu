//! Custom material for displaying noise texture

use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::sprite::Material2d;
use crate::noise::generate_noise_texture;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct NoiseDisplayMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub noise_texture: Handle<Image>,
}

impl Material2d for NoiseDisplayMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/noise_display.wgsl".into()
    }
}

pub fn setup_noise_display(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<NoiseDisplayMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let noise_image = generate_noise_texture();
    let noise_handle = images.add(noise_image);

    let material = NoiseDisplayMaterial {
        noise_texture: noise_handle.clone(),
    };
    let material_handle = materials.add(material);

    // Use Bevy's built-in Rectangle mesh which has proper UVs
    let quad = Mesh::from(Rectangle::new(1280.0, 720.0));
    let mesh_handle = meshes.add(quad);

    commands.spawn((
        Mesh2d(mesh_handle),
        MeshMaterial2d(material_handle),
        Transform::default(),
    ));

    commands.spawn((
        Camera2d,
        Transform::default(),
    ));
}