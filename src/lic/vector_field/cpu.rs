//! CPU-backed vector field node

use bevy::prelude::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::render_resource::*;
use bevy::render::render_resource::RawComputePipelineDescriptor;
use crate::lic::vector_field::VectorFieldNode;

/// Vector field node that uploads CPU pixel-space data and converts to UV-space via compute shader
pub struct CpuVectorFieldNode {
    pub width: u32,
    pub height: u32,
    /// Interleaved (dx_px, dy_px) per pixel in pixel-space
    pub pixel_data: Vec<f32>,
    
    // Internal GPU resources
    staging_buffer: Option<Buffer>,
    raw_texture: Option<Texture>,
    cooked_texture: Option<Texture>,
    cooked_view: Option<TextureView>,
    bind_group: Option<BindGroup>,
    pipeline: Option<ComputePipeline>,
    uv_per_pixel_buffer: Option<Buffer>,
    initialized: bool,
}

impl CpuVectorFieldNode {
    pub fn new(width: u32, height: u32) -> Self {
        let pixel_count = (width * height) as usize;
        Self {
            width,
            height,
            pixel_data: vec![0.0; pixel_count * 2],
            staging_buffer: None,
            raw_texture: None,
            cooked_texture: None,
            cooked_view: None,
            bind_group: None,
            pipeline: None,
            uv_per_pixel_buffer: None,
            initialized: false,
        }
    }
    
    fn initialize(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        if self.initialized {
            return;
        }
        
        // Staging buffer for CPU -> GPU upload
        let staging_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("vector_field_staging"),
            size: (self.pixel_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Raw texture (pixel-space RG16Float)
        let raw_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("vector_field_raw"),
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        // Cooked texture (UV-space RG16Float)
        let cooked_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("vector_field_cooked"),
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let cooked_view = cooked_texture.create_view(&TextureViewDescriptor::default());
        
        // Compute pipeline for pixel-space to UV-space conversion
        let shader = render_device.create_shader_module(ShaderModuleDescriptor {
            label: Some("vecfield_px_to_uv"),
            source: ShaderSource::Wgsl(include_str!("../shaders/vecfield_px_to_uv.wgsl").into()),
        });
        
        let bind_group_layout = render_device.create_bind_group_layout(
            "vecfield_px_to_uv_bind_group_layout",
            &[
                // Raw texture (pixel-space)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Cooked texture (UV-space)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rg16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Uniform: uv_per_pixel (1/width, 1/height)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(8).unwrap()),
                    },
                    count: None,
                },
            ],
        );
        
        let pipeline_layout = render_device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label: Some("vecfield_px_to_uv_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            },
        );
        
        let pipeline = render_device.create_compute_pipeline(
            &RawComputePipelineDescriptor {
                label: Some("vecfield_px_to_uv_pipeline".into()),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main".into(),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        
        // Uniform buffer for uv_per_pixel
        let uv_per_pixel_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("uv_per_pixel_uniform"),
            size: 8,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let uv_per_pixel = [1.0 / self.width as f32, 1.0 / self.height as f32];
        render_queue.write_buffer(&uv_per_pixel_buffer, 0, bytemuck::cast_slice(&uv_per_pixel));
        
        let raw_view = raw_texture.create_view(&TextureViewDescriptor::default());
        
        let bind_group = render_device.create_bind_group(
            "vecfield_px_to_uv_bind_group",
            &bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&raw_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&cooked_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: uv_per_pixel_buffer.as_entire_binding(),
                },
            ],
        );
        
        self.staging_buffer = Some(staging_buffer);
        self.raw_texture = Some(raw_texture);
        self.cooked_texture = Some(cooked_texture);
        self.cooked_view = Some(cooked_view);
        self.bind_group = Some(bind_group);
        self.pipeline = Some(pipeline);
        self.uv_per_pixel_buffer = Some(uv_per_pixel_buffer);
        self.initialized = true;
    }
}

impl VectorFieldNode for CpuVectorFieldNode {
    fn prepare(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        self.initialize(render_device, render_queue);
        
        let staging_buffer = self.staging_buffer.as_ref().unwrap();
        let raw_texture = self.raw_texture.as_ref().unwrap();
        let pipeline = self.pipeline.as_ref().unwrap();
        let bind_group = self.bind_group.as_ref().unwrap();
        
        // Upload pixel data to staging buffer
        render_queue.write_buffer(staging_buffer, 0, bytemuck::cast_slice(&self.pixel_data));
        
        // Copy staging buffer to raw texture
        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("vector_field_upload_encoder"),
        });
        
        encoder.copy_buffer_to_texture(
            ImageCopyBuffer {
                buffer: staging_buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 2 * std::mem::size_of::<f32>() as u32),
                    rows_per_image: Some(self.height),
                },
            },
            ImageCopyTexture {
                texture: raw_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        
        // Dispatch compute shader to convert pixel-space to UV-space
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("vecfield_px_to_uv_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            let workgroup_count = ((self.width + 15) / 16, (self.height + 15) / 16, 1);
            compute_pass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);
        }
        
        render_queue.submit(Some(encoder.finish()));
    }
    
    fn output_view(&self) -> &TextureView {
        self.cooked_view.as_ref().expect("CpuVectorFieldNode not initialized")
    }
    
    fn size(&self) -> UVec2 {
        UVec2::new(self.width, self.height)
    }
}