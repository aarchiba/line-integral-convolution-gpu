//! Perlin noise texture generation on CPU

use bevy::prelude::*;
use noise::{NoiseFn, Perlin};

pub const NOISE_WIDTH: u32 = 512;
pub const NOISE_HEIGHT: u32 = 512;

pub fn generate_noise_grayscale(width: u32, height: u32, scale: f64) -> Vec<f32> {
    let perlin = Perlin::new(42);
    let octaves = 4;
    let persistence = 0.5;
    let lacunarity = 2.0;

    let mut data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let mut value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = scale;
            let mut max_value = 0.0;

            for _ in 0..octaves {
                let nx = x as f64 * frequency;
                let ny = y as f64 * frequency;
                value += amplitude * perlin.get([nx, ny]);
                max_value += amplitude;
                amplitude *= persistence;
                frequency *= lacunarity;
            }

            value /= max_value;
            value = (value + 1.0) * 0.5;
            data.push(value.clamp(0.0, 1.0) as f32);
        }
    }

    data
}

pub fn generate_noise_texture() -> Image {
    let noise = generate_noise_grayscale(NOISE_WIDTH, NOISE_HEIGHT, 0.01);

    let mut data = Vec::with_capacity((NOISE_WIDTH * NOISE_HEIGHT * 4) as usize);

    for value in noise {
        let byte = (value.clamp(0.0, 1.0) * 255.0) as u8;

        data.push(byte);
        data.push(byte);
        data.push(byte);
        data.push(255);
    }

    Image::new(
        bevy::render::render_resource::Extent3d {
            width: NOISE_WIDTH,
            height: NOISE_HEIGHT,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        data,
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::render::render_asset::RenderAssetUsages::RENDER_WORLD,
    )
}