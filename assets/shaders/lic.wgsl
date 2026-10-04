#import bevy_sprite::{
    mesh2d_vertex_output::VertexOutput,
    mesh2d_view_bindings::view,
}

// NOTE: weights are vec4-packed (64 x vec4 = 256 floats) because WGSL uniform
// address space requires array stride 16; array<f32, 256> fails validation.
// Index with forward_weights[i / 4u][i % 4u]. Rust side mirrors this layout
// (see lic/params.rs); logical contract is unchanged (256 + 256 f32 taps).
struct LicParams {
    forward_count: u32,
    backward_count: u32,
    _pad0: u32,
    _pad1: u32,
    forward_weights: array<vec4<f32>, 64>,
    backward_weights: array<vec4<f32>, 64>,
}

@group(2) @binding(0) var noise_texture: texture_2d<f32>;
@group(2) @binding(1) var noise_sampler: sampler;
@group(2) @binding(2) var vector_field_texture: texture_2d<f32>;
@group(2) @binding(3) var vector_field_sampler: sampler;
@group(2) @binding(4) var<uniform> params: LicParams;

@fragment
fn fragment(mesh: VertexOutput) -> @location(0) vec4<f32> {
    // BASIC STREAMLINE INTEGRATION (ticket 5 / issue #6):
    // Sample vector field (UV-space step), walk forward/backward along the
    // streamline, accumulate noise x kernel weight, normalize by weight sum.
    //
    // Kernel contract (see lic/params.rs + LIC_DESIGN.md):
    //   forward_weights[0] = center weight (step 0)
    //   forward_weights[i] = weight for step i forward (i >= 1)
    //   backward_weights[i] = weight for step i+1 backward (i >= 0)
    // Center is sampled exactly once (forward loop); the backward loop starts
    // one step behind the center so it never re-samples it.
    let step_uv = textureSample(vector_field_texture, vector_field_sampler, mesh.uv).xy;

    var acc: f32 = 0.0;
    var wsum: f32 = 0.0;

    // Forward: pos = uv + i * step_uv for i in 0..forward_count
    var pos_f = mesh.uv;
    for (var i: u32 = 0u; i < params.forward_count; i = i + 1u) {
        let n = textureSample(noise_texture, noise_sampler, pos_f).r;
        let w = params.forward_weights[i / 4u][i % 4u];
        acc = acc + n * w;
        wsum = wsum + w;
        pos_f = pos_f + step_uv;
    }

    // Backward: pos = uv - (i+1) * step_uv for i in 0..backward_count
    var pos_b = mesh.uv - step_uv;
    for (var j: u32 = 0u; j < params.backward_count; j = j + 1u) {
        let n = textureSample(noise_texture, noise_sampler, pos_b).r;
        let w = params.backward_weights[j / 4u][j % 4u];
        acc = acc + n * w;
        wsum = wsum + w;
        pos_b = pos_b - step_uv;
    }

    var result: f32 = 0.0;
    if (wsum > 0.0) {
        result = acc / wsum;
    }
    return vec4<f32>(result, result, result, 1.0);
}
