//! LIC material (ticket 5: streamline integration)

use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::sprite::{Material2d, Material2dPlugin};

use crate::lic::params::LicParams;

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