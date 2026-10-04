//! Demo: render LIC output through the real code path and save PNGs.
//!
//! Uses the same Perlin fBm noise as `src/noise.rs`, the analytical fields
//! from `lic::vector_field::analytical`, and the CPU LIC reference from
//! `lic::cpu_reference` (exact mirror of `assets/shaders/lic.wgsl`).
//!
//! Run with: `cargo run --example lic_demo`
//! Output: `target/demo/noise.png`, `target/demo/lic_constant.png`,
//! `target/demo/lic_vortex.png`.

use image::GrayImage;
use line_integral_convolution_gpu::lic::cpu_reference::cpu_lic_image_general;
use line_integral_convolution_gpu::lic::vector_field::analytical;
use line_integral_convolution_gpu::lic::LicParams;
use noise::{NoiseFn, Perlin};

const W: u32 = 256;
const H: u32 = 256;

fn generate_noise() -> Vec<f32> {
    let perlin = Perlin::new(42);
    // Finer grain than the app's display noise so the convolution span
    // covers several noise features and the streaks become visible.
    let scale = 0.06;
    let octaves = 4;
    let persistence = 0.5;
    let lacunarity = 2.0;

    let mut noise = vec![0.0; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let mut value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = scale;
            let mut max_value = 0.0;
            for _ in 0..octaves {
                value += amplitude
                    * perlin.get([(x as f64) * frequency, (y as f64) * frequency]);
                max_value += amplitude;
                amplitude *= persistence;
                frequency *= lacunarity;
            }
            value /= max_value;
            noise[(y * W + x) as usize] = ((value + 1.0) * 0.5).clamp(0.0, 1.0) as f32;
        }
    }
    noise
}

fn save(data: &[f32], name: &str) {
    let mut img = GrayImage::new(W, H);
    for (i, v) in data.iter().enumerate() {
        let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
        let x = (i as u32) % W;
        let y = (i as u32) / W;
        img.put_pixel(x, y, image::Luma([byte]));
    }
    let path = format!("target/demo/{name}");
    img.save(&path).expect("PNG save must succeed");
    println!("wrote {path}");
}

fn main() {
    std::fs::create_dir_all("target/demo").expect("demo dir must be creatable");
    let noise = generate_noise();
    save(&noise, "noise.png");

    // Horizontal streaks: constant field, 1.5 px steps, triangular radius 12
    // (span ~36 px, several noise features wide).
    let field = analytical::constant_horizontal(W, H, 1.5);
    let params = LicParams::triangular_kernel(12);
    let lic = cpu_lic_image_general(&noise, &field, W as usize, H as usize, &params);
    save(&lic, "lic_constant.png");

    // Circular streaks: vortex at the center.
    let c = ((W - 1) as f32) / 2.0;
    let field = analytical::vortex(W, H, (c, c), 60.0);
    let lic = cpu_lic_image_general(&noise, &field, W as usize, H as usize, &params);
    save(&lic, "lic_vortex.png");
}
