@group(0) @binding(0) var raw_texture: texture_2d<f32>;
@group(0) @binding(1) var cooked_texture: texture_storage_2d<rg16float, write>;
@group(0) @binding(2) var<uniform> uv_per_pixel: vec2<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(raw_texture);
    let x = global_id.x;
    let y = global_id.y;
    
    if x >= dimensions.x || y >= dimensions.y {
        return;
    }
    
    let pixel_value = textureLoad(raw_texture, vec2<i32>(x as i32, y as i32));
    let step_px = vec2<f32>(pixel_value.x, pixel_value.y);
    let step_uv = step_px * uv_per_pixel;
    
    textureStore(cooked_texture, vec2<i32>(x as i32, y as i32), vec2<f32>(step_uv.x, step_uv.y));
}