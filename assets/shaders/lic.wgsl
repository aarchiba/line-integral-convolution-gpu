@group(0) @binding(0) var vector_field: texture_2d<f32>;
@group(0) @binding(1) var sampler_linear: sampler;
@group(0) @binding(2) var<uniform> lic_params: LicUniforms;
@group(0) @binding(3) var<uniform> field_params: VectorFieldUniforms;
@group(0) @binding(4) var<uniform> viewport: ViewportUniforms;

struct LicUniforms {
    step_size: f32,
    num_steps: u32,
    noise_scale: f32,
    contrast: f32,
    _padding: vec4<f32>,
}

struct VectorFieldUniforms {
    scale: f32,
    offset: vec2<f32>,
    _padding: vec2<f32>,
}

struct ViewportUniforms {
    inv_size: vec2<f32>,
    _padding: vec2<f32>,
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy * viewport.inv_size;
    let field_uv = uv * field_params.scale + field_params.offset;

    // Line Integral Convolution
    var accumulated = 0.0;
    var weight_sum = 0.0;
    var pos = field_uv;

    let half_steps = lic_params.num_steps / 2;

    // Forward integration
    for (var i = 0u; i < half_steps; i++) {
        let field = textureSample(vector_field, sampler_linear, pos).rg;
        let v = field; // Already in [-1, 1] range

        let sample_uv = uv + v * lic_params.step_size * f32(i);
        let sample = textureSample(vector_field, sampler_linear, sample_uv).r;

        let weight = 1.0 - f32(i) / f32(half_steps);
        accumulated += sample * weight;
        weight_sum += weight;

        pos += v * lic_params.step_size;

        // Early exit if we leave the domain
        if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0) {
            break;
        }
    }

    // Backward integration
    pos = field_uv;
    for (var i = 0u; i < half_steps; i++) {
        let field = textureSample(vector_field, sampler_linear, pos).rg;
        let v = field;

        let sample_uv = uv - v * lic_params.step_size * f32(i);
        let sample = textureSample(vector_field, sampler_linear, sample_uv).r;

        let weight = 1.0 - f32(i) / f32(half_steps);
        accumulated += sample * weight;
        weight_sum += weight;

        pos -= v * lic_params.step_size;

        if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0) {
            break;
        }
    }

    let result = accumulated / max(weight_sum, 0.001);
    let contrasted = pow(result, lic_params.contrast);

    return vec4<f32>(vec3<f32>(contrasted), 1.0);
}