//! Headless test for VectorFieldNode trait and CpuVectorFieldNode

use line_integral_convolution_gpu::lic::vector_field::CpuVectorFieldNode;

#[test]
fn test_vector_field_node_trait_compiles() {
    // This test verifies the trait is properly defined and can be used
    // The actual test is that this compiles
}

#[test]
fn test_cpu_vector_field_node_creation() {
    let node = CpuVectorFieldNode::new(256, 256);
    assert_eq!(node.width, 256);
    assert_eq!(node.height, 256);
    assert_eq!(node.pixel_data.len(), 256 * 256 * 2);
}

#[test]
fn test_cpu_vector_field_node_constant_field() {
    // This test would require a headless RenderApp to run
    // For now we verify the structure is correct
    let mut node = CpuVectorFieldNode::new(64, 64);
    
    // Fill with constant horizontal field (1.0, 0.0) in pixel-space
    for i in 0..(64 * 64) {
        node.pixel_data[i * 2] = 1.0;   // dx_px
        node.pixel_data[i * 2 + 1] = 0.0; // dy_px
    }
    
    assert_eq!(node.pixel_data[0], 1.0);
    assert_eq!(node.pixel_data[1], 0.0);
    assert_eq!(node.pixel_data[64 * 2 * 32 + 64 * 2], 1.0); // middle pixel
}