# Line Integral Convolution on GPU

A Rust/Bevy/wgpu implementation of Line Integral Convolution (LIC) for visualizing vector fields on the GPU.

## Overview

Line Integral Convolution is a technique for visualizing vector fields by convolving a noise texture along streamlines of the field. This implementation:

- Generates procedural vector fields using Perlin noise
- Performs LIC integration entirely in a WGSL fragment shader
- Uses Bevy's render graph for efficient GPU rendering
- Supports hot-reloading of shaders during development

## Project Structure

```
projects/line-integral-convolution-gpu/
├── Cargo.toml              # Rust dependencies and config
├── src/
│   └── main.rs             # Main application entry point
├── assets/
│   ├── shaders/
│   │   └── lic.wgsl        # WGSL shader for LIC computation
│   └── meshes/
│       └── lic_quad.mesh   # Fullscreen quad mesh asset
└── README.md               # This file
```

## Dependencies

- **Bevy 0.15** - Game engine with ECS and wgpu-based renderer
- **wgpu 24** - Native WebGPU implementation
- **noise 0.8** - Procedural noise generation (Perlin)
- **rand / rand_chacha** - Random number generation
- **bytemuck** - Zero-copy serialization for GPU uniforms

## Building & Running

```bash
cargo run --release
```

## Controls (Planned)

- Mouse drag: Pan vector field
- Scroll: Zoom in/out
- Keys 1-9: Switch vector field presets
- Space: Pause/resume animation
- R: Regenerate noise seed

## Shader Hot-Reload

Edit `assets/shaders/lic.wgsl` while the application is running to see changes instantly (requires `dynamic_linking` feature on Bevy).

## LIC Algorithm

The fragment shader performs bidirectional streamline integration:

1. Sample vector field at current position
2. Step along/against the vector direction
3. Accumulate noise texture samples with triangular weighting
4. Normalize by accumulated weight
5. Apply contrast adjustment

## Vector Field Generation

Currently uses 2D Perlin noise to generate a swirling vector field. Future enhancements:

- Fluid simulation (Navier-Stokes)
- Magnetic field visualization
- User-defined analytic fields
- Time-varying animated fields

## License

MIT