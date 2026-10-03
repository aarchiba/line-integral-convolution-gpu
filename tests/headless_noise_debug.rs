//! Headless test to debug noise texture rendering

use bevy::render::render_resource::*;
use line_integral_convolution_gpu::noise::generate_noise_texture;

#[test]
fn test_noise_texture_generation() {
    let image = generate_noise_texture();
    
    // Verify texture properties
    assert_eq!(image.width(), 512);
    assert_eq!(image.height(), 512);
    assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba8UnormSrgb);
    
    // Verify data is not all zeros - use the data field directly
    let data = &image.data;
    let non_zero_count = data.iter().filter(|&&b| b != 0).count();
    println!("Non-zero bytes: {} out of {}", non_zero_count, data.len());
    assert!(non_zero_count > 0, "Texture should have non-zero data");
    
    // Check statistical properties
    let sum: u32 = data.iter().map(|&b| b as u32).sum();
    let mean = sum as f32 / data.len() as f32;
    println!("Mean pixel value: {}", mean);
    // For normalized noise [0, 255], mean should be around 127
    assert!((mean - 127.0).abs() < 50.0, "Mean should be around 127, got {}", mean);
}

#[test]
fn test_noise_texture_first_pixels() {
    let image = generate_noise_texture();
    let data = &image.data;
    
    // Print first 16 pixels (4 pixels * 4 channels)
    println!("First 16 bytes: {:?}", &data[0..16]);
    
    // Check if there's variation
    let first_pixel = &data[0..4];
    let second_pixel = &data[4..8];
    println!("Pixel 0: {:?}, Pixel 1: {:?}", first_pixel, second_pixel);
}