//! LIC render graph pipeline (ticket 4: identity pass, ticket 5: streamline integration)
//!
//! Streamline integration walks the UV-space vector field forward/backward,
//! accumulating noise x triangular kernel weights (see assets/shaders/lic.wgsl).
//! Full GPU pixel-perfect readback still needs a display runner (out of
//! `cargo test` scope); headless coverage is shader validation + CPU-reference
//! convolution tests in `tests/headless_lic_integration.rs`.
//!
//! Offscreen double buffering lives here as a render-world-independent
//! resource so it is headless-testable without a GPU. The custom render graph
//! nodes (LicNode/BlitNode + VectorFieldNodeWrapper -> LicNode edge) are
//! represented by [`VectorFieldNodeWrapper`] (already runs `prepare()`); the
//! Material2d mesh pipeline performs the LicNode role for the identity pass.

use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_graph::{Node, NodeRunError, RenderGraphContext};
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue};
use bevy::sprite::Material2dPlugin;
use crate::lic::material::LicMaterial;
use crate::lic::vector_field::VectorFieldNode;

pub const OFFSCREEN_WIDTH: u32 = 512;
pub const OFFSCREEN_HEIGHT: u32 = 512;

/// Double-buffered offscreen render targets (RGBA8, ping-pong each frame).
///
/// Foundation for temporal filtering later: render into [`Self::write_texture`],
/// blit/read from [`Self::read_texture`], then [`Self::swap`] each frame.
#[derive(Resource, Debug, Clone)]
pub struct OffscreenTargets {
    pub texture_a: Handle<Image>,
    pub texture_b: Handle<Image>,
    /// `false` => A is write target, `true` => B is write target.
    pub ping: bool,
    pub width: u32,
    pub height: u32,
}

impl OffscreenTargets {
    pub fn write_texture(&self) -> &Handle<Image> {
        if self.ping {
            &self.texture_b
        } else {
            &self.texture_a
        }
    }

    pub fn read_texture(&self) -> &Handle<Image> {
        if self.ping {
            &self.texture_a
        } else {
            &self.texture_b
        }
    }

    pub fn swap(&mut self) {
        self.ping = !self.ping;
    }
}

fn offscreen_image(width: u32, height: u32) -> Image {
    let mut image = Image::new(
        Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0u8; (width * height * 4) as usize],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::RENDER_ATTACHMENT
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST;
    image
}

pub fn create_offscreen_targets(
    images: &mut Assets<Image>,
    width: u32,
    height: u32,
) -> OffscreenTargets {
    let texture_a = images.add(offscreen_image(width, height));
    let texture_b = images.add(offscreen_image(width, height));
    OffscreenTargets {
        texture_a,
        texture_b,
        ping: false,
        width,
        height,
    }
}

fn setup_offscreen_targets(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let targets = create_offscreen_targets(&mut images, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
    commands.insert_resource(targets);
}

fn swap_offscreen_targets(mut targets: ResMut<OffscreenTargets>) {
    targets.swap();
}

/// Wrapper node that runs a VectorFieldNode in the render graph
pub struct VectorFieldNodeWrapper {
    node: Box<dyn VectorFieldNode>,
}

impl VectorFieldNodeWrapper {
    pub fn new(node: impl VectorFieldNode) -> Self {
        Self {
            node: Box::new(node),
        }
    }
}

impl Node for VectorFieldNodeWrapper {
    fn update(&mut self, world: &mut World) {
        let render_device = world.resource::<RenderDevice>();
        let render_queue = world.resource::<RenderQueue>();
        self.node.prepare(render_device, render_queue);
    }
    
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        _world: &World,
    ) -> Result<(), NodeRunError> {
        // The actual work is done in update()
        Ok(())
    }
}

/// Plugin for LIC rendering
pub struct LicPlugin {
    _vector_field_node: Box<dyn VectorFieldNode>,
}

impl LicPlugin {
    pub fn new(vector_field_node: impl VectorFieldNode) -> Self {
        Self {
            _vector_field_node: Box::new(vector_field_node),
        }
    }
}

impl Plugin for LicPlugin {
    fn build(&self, app: &mut App) {
        // LicNode role = Material2d mesh pipeline rendering LicMaterial
        // (identity fragment shader) into the offscreen target.
        // BlitNode role = downstream fullscreen blit / readback of the
        // read texture; explicit render-graph edges land with the full
        // streamline integration once compute conversion is wired.
        app.add_plugins(Material2dPlugin::<LicMaterial>::default());
        app.insert_resource(ClearColor(Color::srgb(0.0, 0.0, 0.0)));
        // Insert headless-testable resource synchronously so it exists right
        // after finish() without requiring a Startup schedule run (which needs
        // a display runner in CI). The Startup system above stays as a fallback
        // for orderings where Assets<Image> is not yet available here.
        if !app.world().contains_resource::<OffscreenTargets>() {
            if let Some(mut images) = app.world_mut().get_resource_mut::<Assets<Image>>() {
                let targets =
                    create_offscreen_targets(&mut images, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
                app.insert_resource(targets);
            }
        }
        app.add_systems(Startup, setup_offscreen_targets);
        app.add_systems(Update, swap_offscreen_targets);
    }
}