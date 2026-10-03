# Glossary

## LIC (Line Integral Convolution)

A texture-based vector field visualization technique. For each output pixel, integrates along the vector field streamline in both directions, sampling a noise texture and accumulating with a weighting kernel.

## Noise Texture

Input texture providing the "ink" for LIC. Typically grayscale Perlin noise. Sampled along streamlines.

## Vector Field Texture

2D texture encoding vector direction and magnitude per pixel. Format: RG16Float (two half-floats = x, y components in UV space [0,1]).

**Critical:** Always stores UV-space step vectors, never pixel-space. Conversion happens once in production pipeline.

## VectorFieldNode (trait)

Abstraction for vector field production. Three implementations:
- `CpuVectorFieldNode`: CPU pixel-space data → upload → compute (px→UV) → cooked texture
- `GpuComputeVectorFieldNode`: User compute shader (pixel-space) → compute (px→UV) → cooked texture
- `PrecookedVectorFieldNode`: Zero-copy pass-through of existing UV-space texture view

## Weighting Kernel (Kernel)

Two 1D arrays in uniform buffer:
- `forward[0]` = center weight (step 0)
- `forward[i]` = weight for step `i` forward from center (i ≥ 1)
- `backward[i]` = weight for step `i+1` backward from center (i ≥ 0)
Explicit counts: `forward_count` (includes center), `backward_count`. MAX_STEPS = 256.

## Streamline Integration

For each output pixel: start at pixel center, step along vector field direction (forward/backward), accumulate `noise(sample_pos) * weight`, normalize by sum of weights.

## Uniform Buffer

GPU buffer bound as `uniform` in WGSL, updated per-frame from CPU. Small (~64KB max), fast access. Used for kernel weights.

## Material2d

Bevy 2D rendering trait for custom fragment shaders. Handles bind groups: 0=view_proj, 1=transform, 2+=material-specific (textures, samplers, uniforms).

## AsBindGroup

Bevy derive macro that generates bind group layout from struct fields annotated with `#[texture]`, `#[sampler]`, `#[uniform]`.

## Bind Group Layout (for LIC Material)

- Group 0: ViewProj uniform (Camera2d)
- Group 1: Transform uniform (quad transform)
- Group 2: noise_texture (texture_2d<f32>) + noise_sampler
- Group 2: vector_field_texture (texture_2d<f32>) + vector_field_sampler
- Group 3: LicParams uniform (forward_count, backward_count, forward_weights[256], backward_weights[256])

## LicParams

Uniform struct passed to LIC shader (matches LIC_DESIGN.md):
- `forward_count: u32` - number of valid entries in forward_weights (includes center at index 0)
- `backward_count: u32` - number of valid entries in backward_weights
- `forward_weights: array<f32, 256>` - forward_weights[0] = center weight, forward_weights[i] = weight for step i forward
- `backward_weights: array<f32, 256>` - backward_weights[i] = weight for step i+1 backward

**Design note:** No `step_size`, no `kernel_center`. Vector field texture stores UV-space step vectors directly. Center is always `forward_weights[0]`.