#import bevy_sprite::{
    mesh2d_vertex_output::VertexOutput,
    mesh2d_view_bindings::view,
}

@group(2) @binding(0) var noise_texture: texture_2d<f32>;
@group(2) @binding(1) var noise_sampler: sampler;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    let color = textureSample(noise_texture, noise_sampler, mesh.uv);
    return color;
}