# LIC Shader Design

## Overview

Line Integral Convolution (LIC) implemented as a single-pass fragment shader using Bevy's Material2d with an offscreen render target. The design separates **production** of the vector field texture from **consumption** by the LIC shader, enabling both CPU and GPU workflows without extra copies.

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VECTOR FIELD PRODUCTION                          │
│                                                                         │
│   ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐   │
│   │ CPU Data     │     │ GPU Compute      │     │ Pre-existing     │   │
│   │ (pixel-space)│     │ (pixel-space)    │     │ UV-space Texture │   │
│   └──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘   │
│          │                      │                        │             │
│          ▼                      ▼                        ▼             │
│   ┌──────────────────────────────────────────────────────────────┐    │
│   │ VectorFieldNode (trait)                                      │    │
│   │   - CpuVectorFieldNode: upload → compute (px→UV) → texture   │    │
│   │   - GpuComputeVectorFieldNode: compute → compute (px→UV)     │    │
│   │   - PrecookedVectorFieldNode: zero-copy pass-through         │    │
│   └──────────────────────────┬───────────────────────────────────┘    │
│                              │                                         │
│                              ▼                                         │
│                    ┌───────────────────┐                               │
│                    │ UV-space RG16Float │  ◄── Single canonical form  │
│                    │ Texture (cooked)   │                               │
│                    └─────────┬─────────┘                               │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LIC CONSUMPTION (Fragment Shader)                │
│                                                                         │
│   ┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐   │
│   │ Noise Texture │     │ Vector Field     │     │ Kernel Weights  │   │
│   │ (RGBA8)       │     │ Texture (RG16F)  │     │ (Uniform Buffer)│   │
│   └──────┬────────┘     └────────┬─────────┘     └────────┬────────┘   │
│          │                       │                        │             │
│          └───────────────────────┼────────────────────────┘             │
│                                  ▼                                     │
│                     ┌────────────────────────┐                         │
│                     │ Fragment Shader (LIC)  │                         │
│                     │                        │                         │
│                     │ For each pixel:        │                         │
│                     │   1. Sample vector     │                         │
│                     │   2. Step along stream │                         │
│                     │   3. Accumulate noise  │                         │
│                     │      × kernel weight   │                         │
│                     │   4. Normalize         │                         │
│                     │   5. Output RGBA       │                         │
│                     └───────────┬────────────┘                         │
│                                 │                                      │
│                                 ▼                                      │
│                     ┌────────────────────────┐                         │
│                     │ Render Target (RGBA8)  │                         │
│                     │ (offscreen texture)    │                         │
│                     └───────────┬────────────┘                         │
│                                 │                                      │
│                     ┌───────────┴───────────┐                          │
│                     ▼                       ▼                          │
│             ┌─────────────┐         ┌─────────────┐                   │
│             │ Blit to     │         │ Read back   │                   │
│             │ Screen      │         │ (optional)  │                   │
│             └─────────────┘         └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Precision Model

| Data | Storage Format | Shader Computation |
|------|----------------|-------------------|
| Noise texture | RGBA8 (8-bit) | Sampled → `f32` [0,1] |
| Vector field | RG16Float (2×16-bit) | Sampled → `f32` |
| Kernel weights | Uniform buffer (`f32[]`) | `f32` |
| Accumulation | Local `f32` variables | Full 32-bit float precision |
| Output | Render target (RGBA8) | `vec4<f32>` → auto-convert |

**Key insight:** Accumulation precision is independent of texture format. Shader always computes in `f32`.

## Kernel Design

Two separate arrays in uniform buffer:
- `forward[0]` = center weight (step 0)
- `forward[i]` = weight for step `i` forward from center (i ≥ 1)
- `backward[i]` = weight for step `i+1` backward from center (i ≥ 0)

Explicit counts passed as uniforms (avoids iterating unused elements):
- `forward_count` = number of valid entries in `forward` (includes center)
- `backward_count` = number of valid entries in `backward`

`MAX_STEPS = 256` (configurable, 2 KB total for both arrays).

**Scale-invariance:** output is `acc / wsum`, so scaling all weights by a constant cancels out — only relative weights matter, there is no need to pre-normalize the kernel to sum to one.

## Boundary Handling

Per-axis sampler-native modes (`ClampToEdge` default, `Repeat`, `MirrorRepeat` via Bevy `ImageSamplerDescriptor`), applied identically to noise and vector-field sampling. No `Zero` mode (see `docs/adr/0001-sampler-native-boundary-modes.md`). Because every tap resolves in-bounds, `wsum` is kernel-constant.

## Uniform Buffer (LicParams)

```wgsl
struct LicParams {
    forward_count: u32,
    backward_count: u32,
    forward_weights: array<f32, 256>,
    backward_weights: array<f32, 256>,
}
```

No `step_size`, no `kernel_center` - both eliminated by design:
- Vector field texture stores UV-space step vectors directly (no scaling in shader)
- Center is always `forward_weights[0]`

## Vector Field Texture

**Format:** RG16Float (two half-floats = step_x, step_y in UV space [0,1])

**Critical:** The vector field texture **always** stores UV-space step vectors, never pixel-space. The conversion from pixel-space to UV-space happens exactly once in the production pipeline, not in the LIC shader.

### Production Pipeline (VectorFieldNode)

The `VectorFieldNode` trait abstracts vector field production. Three built-in implementations:

#### 1. CpuVectorFieldNode
- Caller provides pixel-space steps `(dx_px, dy_px)` per frame
- Node uploads to staging buffer → raw texture (pixel-space)
- Compute shader: `step_uv = step_px * uv_per_pixel` → cooked texture (UV-space)
- One compute dispatch per frame, no CPU-side conversion

#### 2. GpuComputeVectorFieldNode
- User provides a compute shader node that writes pixel-space vectors to a texture
- Node runs user's compute → raw texture (pixel-space)
- Same conversion compute shader → cooked texture (UV-space)
- Zero CPU involvement after setup

#### 3. PrecookedVectorFieldNode
- Caller already has UV-space texture on GPU (from another render pass, external source)
- Zero-copy: passes the `TextureView` directly to LIC material
- `prepare()` is a no-op

## Bind Group Layout

| Group | Binding | Resource |
|-------|---------|----------|
| 0 | 0 | ViewProj uniform (Camera2d) |
| 1 | 0 | Transform uniform (quad) |
| 2 | 0 | Noise texture (`texture_2d<f32>`) |
| 2 | 1 | Noise sampler |
| 2 | 2 | Vector field texture (`texture_2d<f32>`) |
| 2 | 3 | Vector field sampler |
| 3 | 0 | LicParams uniform buffer |

The material binds **texture views** (`GpuTextureView`), not Bevy `Handle<Image>`. This allows any GPU texture source.

## Shader Entry Points

```wgsl
@vertex fn vs_main(...) -> VertexOutput { ... }
@fragment fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> { ... }
```

## Shader File Locations

Shaders live in two places, split by loading mechanism:

- `assets/shaders/*.wgsl` — **Bevy asset shaders** (`noise_display.wgsl`,
  `lic.wgsl`). Referenced by path from `Material2d::fragment_shader()`
  (`"shaders/xxx.wgsl"`). They go through Bevy's shader preprocessor (hence
  `#import bevy_sprite::...`), get hot-reload, and are validated at render-world
  extraction time.
- `src/lic/shaders/*.wgsl` — **inline shaders** (`vecfield_px_to_uv.wgsl`).
  Loaded with `include_str!` in `vector_field/cpu.rs` and compiled directly via
  `RenderDevice::create_shader_module`. No `#import` support, no hot-reload —
  but the WGSL is versioned alongside the Rust code that builds its bind-group
  layout, so the two can't drift.

## Public API (Rust Module)

```
lic/
├── mod.rs                    # Public API, LicPlugin
├── params.rs                 # LicParams struct
├── material.rs               # LicMaterial (Material2d impl)
├── vector_field/
│   ├── mod.rs                # VectorFieldNode trait + implementations
│   ├── cpu.rs                # CpuVectorFieldNode
│   ├── gpu_compute.rs        # GpuComputeVectorFieldNode
│   └── precooked.rs          # PrecookedVectorFieldNode
├── pipeline.rs               # Render graph setup, LicNode
└── shaders/
    ├── lic.wgsl              # LIC fragment shader
    └── vecfield_px_to_uv.wgsl # Pixel→UV conversion compute shader
```

```rust
// ===== Core types =====
pub struct LicParams {
    pub forward_count: u32,
    pub backward_count: u32,
    pub forward_weights: [f32; 256],
    pub backward_weights: [f32; 256],
}

/// Lightweight handle to a GPU texture view (not a Bevy Asset)
pub struct GpuTextureView {
    pub view: TextureView,
    pub format: TextureFormat,
    pub size: UVec2,
}

// ===== Vector Field Production =====
pub trait VectorFieldNode: Send + Sync + 'static {
    fn prepare(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue);
    fn output_view(&self) -> &TextureView;
    fn format(&self) -> TextureFormat { TextureFormat::Rg16Float }
    fn size(&self) -> UVec2;
}

pub struct CpuVectorFieldNode {
    pub width: u32,
    pub height: u32,
    pub pixel_data: Vec<f32>,  // (dx_px, dy_px) interleaved
    // ... internal GPU resources
}

pub struct GpuComputeVectorFieldNode {
    pub compute_node: Box<dyn ComputeNode>,
    // ... internal GPU resources
}

pub struct PrecookedVectorFieldNode {
    pub texture_view: TextureView,
}

// ===== Integration =====
pub struct LicPlugin {
    vector_field_node: Box<dyn VectorFieldNode>,
}

impl LicPlugin {
    pub fn new(vector_field_node: impl VectorFieldNode) -> Self { ... }
}

// Per-frame kernel update
pub fn update_lic_kernel(params: &mut LicParams, time: f32) { ... }
```

## Integration Steps

1. **Choose vector field source** and create appropriate `VectorFieldNode`
2. **Startup**: `LicPlugin::new(vector_field_node)` registers render graph nodes
3. **Per-frame**:
   - Update `LicParams` uniform (kernel weights, counts)
   - Update noise texture if animated (via `GpuTextureView` or Bevy asset)
   - Update vector field:
     - `CpuVectorFieldNode`: modify `pixel_data` in a system
     - `GpuComputeVectorFieldNode`: user's compute node runs automatically
     - `PrecookedVectorFieldNode`: swap `texture_view` if needed
   - Render graph executes: VectorFieldNode → LicNode → output

## Render Graph Structure

```
CoreRenderGraphNode::Prepass
    │
    ├─► VectorFieldNodeWrapper (runs selected VectorFieldNode.prepare())
    │       └─► Produces cooked vector field texture view
    │
    └─► LicNode (Material2d render)
            └─► Reads cooked vector field texture view
            └─► Writes to offscreen render target
```

Ordering enforced by render graph edges: `VectorFieldNodeWrapper` → `LicNode`.

## Dynamic Texture Updates

### Noise Texture (RGBA8)
- **Use case**: Animated noise, different noise patterns per frame, time-varying "ink"
- **Update method**: Create `GpuTextureView` from compute shader output, or use Bevy `Image` asset with `RenderAssetUsages::RENDER_WORLD`
- **Frequency**: Every frame if needed

### Vector Field Texture (RG16Float, UV-space)
- **Handled entirely by the selected `VectorFieldNode`**:
  - `CpuVectorFieldNode`: Update `pixel_data` each frame; node handles upload + conversion compute
  - `GpuComputeVectorFieldNode`: User's compute shader runs each frame; node runs conversion compute
  - `PrecookedVectorFieldNode`: Swap `texture_view` reference; zero GPU work
- **No extra copies**: Conversion compute writes directly to final texture bound by LIC material
- **Frequency**: Every frame for interactive/animated fields

## Future Extensions

- **Compute shader path**: Replace fragment shader with compute for ping-pong temporal filtering
- **Multiple kernels**: Different weights per channel (RGB = different scales)
- **Adaptive step size**: Based on vector field magnitude
- **Export**: Read back render target via `RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD`
- **Custom VectorFieldNode implementations**: User-defined production pipelines (e.g., video decode → vector field, ML inference → vector field)