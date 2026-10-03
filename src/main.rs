use bevy::{
    asset::{load_internal_asset},
    core_pipeline::{
        core_2d::graph::{Core2d, Node2d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    },
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_graph::{NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner},
        render_phase::DrawError,
        render_resource::{
            BindGroup, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, FragmentState, MultisampleState, Operations, PipelineCache,
            PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor,
            Shader, ShaderStages, ShaderType, SpecializedRenderPipeline, SpecializedRenderPipelines,
            TextureFormat, TextureSampleType,
            BufferUsages, BufferInitDescriptor,
            BindingResource, BindGroupEntry,
        },
        renderer::{RenderContext, RenderDevice},
        texture::GpuImage,
        view::ViewTarget,
        Extract, ExtractSchedule, Render, RenderApp, RenderSet,
    },
    window::{PresentMode, WindowResolution},
};
use bevy::render::render_resource::binding_types::{texture_2d, sampler, uniform_buffer};
use bevy::ecs::system::SystemParamItem;

use bytemuck::{Pod, Zeroable, cast_slice};
use noise::{NoiseFn, Perlin};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const LIC_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(0x123456789abcdef0123456789abcdef0);

// ============================================================================
// Components & Resources
// ============================================================================

#[derive(Resource, Clone, Default)]
struct VectorFieldTexture {
    image: Option<Handle<Image>>,
}

#[derive(Resource)]
struct LicParams {
    step_size: f32,
    num_steps: u32,
    noise_scale: f32,
    contrast: f32,
}

impl Default for LicParams {
    fn default() -> Self {
        Self {
            step_size: 0.003,
            num_steps: 40,
            noise_scale: 1.0,
            contrast: 1.0,
        }
    }
}

// ============================================================================
// Uniforms
// ============================================================================

#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Default, Resource)]
#[repr(C)]
struct LicUniforms {
    step_size: f32,
    num_steps: u32,
    noise_scale: f32,
    contrast: f32,
    _padding: [f32; 4],
}

#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Default, Resource)]
#[repr(C)]
struct VectorFieldUniforms {
    scale: f32,
    _pad1: f32,
    offset: Vec2,
    _padding: [f32; 2],
}

#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Default, Resource)]
#[repr(C)]
struct ViewportUniforms {
    inv_size: Vec2,
    _padding: Vec2,
}

// ============================================================================
// Pipeline
// ============================================================================

#[derive(Resource)]
struct LicPipeline {
    layout: BindGroupLayout,
    pipeline_id: CachedRenderPipelineId,
    bind_group: BindGroup,
}

impl SpecializedRenderPipeline for LicPipeline {
    type Key = ();

    fn specialize(&self, _key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("lic_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: LIC_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "fs_main".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::bevy_default(),
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        }
    }
}

// ============================================================================
// Extract
// ============================================================================

fn extract_lic_params(
    mut commands: Commands,
    params: Extract<Res<LicParams>>,
) {
    commands.insert_resource(LicUniforms {
        step_size: params.step_size,
        num_steps: params.num_steps,
        noise_scale: params.noise_scale,
        contrast: params.contrast,
        _padding: [0.0; 4],
    });
}

fn extract_vector_field(
    mut commands: Commands,
    vector_field: Extract<Res<VectorFieldTexture>>,
) {
    commands.insert_resource(vector_field.clone());
}

fn extract_field_uniforms(
    mut commands: Commands,
    field_uniforms: Extract<Res<VectorFieldUniforms>>,
) {
    commands.insert_resource(field_uniforms.clone());
}

fn extract_viewport(
    mut commands: Commands,
    windows: Extract<Query<&Window>>,
) {
    if let Ok(window) = windows.get_single() {
        let inv_size = Vec2::new(1.0 / window.width(), 1.0 / window.height());
        commands.insert_resource(ViewportUniforms { inv_size, _padding: Vec2::ZERO });
    }
}

// ============================================================================
// Prepare
// ============================================================================

fn prepare_lic_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    render_device: Res<RenderDevice>,
    vector_field: Res<VectorFieldTexture>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    lic_params: Res<LicUniforms>,
    field_params: Res<VectorFieldUniforms>,
    viewport: Res<ViewportUniforms>,
) {
    let Some(field_handle) = &vector_field.image else { return };
    let Some(gpu_image) = gpu_images.get(field_handle) else { return };

    let layout = render_device.create_bind_group_layout(
        "lic_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<LicUniforms>(false),
                uniform_buffer::<VectorFieldUniforms>(false),
                uniform_buffer::<ViewportUniforms>(false),
            ),
        ),
    );

    let lic_uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("lic_uniform_buffer"),
        contents: cast_slice(&[*lic_params]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let field_uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("field_uniform_buffer"),
        contents: cast_slice(&[*field_params]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let viewport_uniform_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("viewport_uniform_buffer"),
        contents: cast_slice(&[*viewport]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    let bind_group = render_device.create_bind_group(
        "lic_bind_group",
        &layout,
        &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&gpu_image.texture_view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(&sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: lic_uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: field_uniform_buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: viewport_uniform_buffer.as_entire_binding(),
            },
        ],
    );

    let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("lic_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_shader_vertex_state(),
        fragment: Some(FragmentState {
            shader: LIC_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "fs_main".into(),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::bevy_default(),
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        push_constant_ranges: vec![],
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(LicPipeline { layout, pipeline_id, bind_group });
}

// ============================================================================
// Render Graph Node
// ============================================================================

#[derive(Default)]
struct LicNode;

impl ViewNode for LicNode {
    type ViewQuery = ();

    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        _view_query: SystemParamItem<'_, '_, Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let lic_pipeline = world.resource::<LicPipeline>();
        
        let Some(pipeline) = pipeline_cache.get_render_pipeline(lic_pipeline.pipeline_id) else {
            return Ok(());
        };

        let view_entity = graph.view_entity();
        let view_target = world.get_entity(view_entity)
            .ok()
            .and_then(|e| e.get::<ViewTarget>())
            .ok_or(NodeRunError::DrawError(DrawError::RenderCommandFailure("ViewTarget not found")))?;

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("lic_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: view_target.main_texture_view(),
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &lic_pipeline.bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

// ============================================================================
// Setup
// ============================================================================

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut vector_field: ResMut<VectorFieldTexture>,
) {
    let field_size: u32 = 512;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let perlin = Perlin::new(rng.gen());

    let mut field_data = vec![0.0f32; (field_size * field_size * 4) as usize];

    for y in 0..field_size {
        for x in 0..field_size {
            let fx = x as f32 / field_size as f32;
            let fy = y as f32 / field_size as f32;

            let angle = perlin.get([fx as f64 * 4.0, fy as f64 * 4.0, 0.0]) * std::f64::consts::TAU;
            let vx = angle.cos() as f32;
            let vy = angle.sin() as f32;

            let idx = ((y * field_size + x) * 4) as usize;
            field_data[idx] = vx;
            field_data[idx + 1] = vy;
            field_data[idx + 2] = 0.0;
            field_data[idx + 3] = 1.0;
        }
    }

    let field_image = Image::new_fill(
        bevy::render::render_resource::Extent3d {
            width: field_size,
            height: field_size,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        bytemuck::cast_slice(&field_data),
        TextureFormat::Rgba32Float,
        bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
    );

    vector_field.image = Some(images.add(field_image));

    commands.spawn((
        Camera2d::default(),
    ));

    commands.insert_resource(VectorFieldUniforms {
        scale: 1.0,
        _pad1: 0.0,
        offset: Vec2::ZERO,
        _padding: [0.0; 2],
    });
}

// ============================================================================
// App Setup
// ============================================================================

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Line Integral Convolution - GPU".into(),
                resolution: WindowResolution::new(1024.0, 1024.0),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(LicParams::default())
        .insert_resource(LicUniforms::default())
        .insert_resource(VectorFieldTexture::default())
        .insert_resource(VectorFieldUniforms::default())
        .insert_resource(ViewportUniforms::default())
        .add_systems(Startup, setup)
        .add_systems(Update, update_lic_uniforms)
        .add_plugins(LicPlugin)
        .run();
}

fn update_lic_uniforms(
    params: Res<LicParams>,
    mut uniforms: ResMut<LicUniforms>,
) {
    uniforms.step_size = params.step_size;
    uniforms.num_steps = params.num_steps;
    uniforms.noise_scale = params.noise_scale;
    uniforms.contrast = params.contrast;
}

// ============================================================================
// Plugin
// ============================================================================

struct LicPlugin;

impl Plugin for LicPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            LIC_SHADER_HANDLE,
            "../assets/shaders/lic.wgsl",
            Shader::from_wgsl
        );

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(ExtractSchedule, (extract_lic_params, extract_vector_field, extract_field_uniforms, extract_viewport))
            .add_systems(Render, prepare_lic_pipeline.in_set(RenderSet::Prepare))
            .add_render_graph_node::<ViewNodeRunner<LicNode>>(Core2d, Node2d::PostProcessing);
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        render_app.init_resource::<SpecializedRenderPipelines<LicPipeline>>();
    }
}