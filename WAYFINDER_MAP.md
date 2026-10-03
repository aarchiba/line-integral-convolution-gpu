# Wayfinder Map: Vector Field Visualization Tool

## Destination

A Rust/Bevy/wgpu tool for visualizing vector fields using Line Integral Convolution (LIC), built incrementally for a Rust beginner who wants to learn graphics concepts. The tool starts from "blue window" → "noise texture" → "LIC shader" → "interactive controls" → "code-defined vortex fields with mouse interaction".

## Notes

- **User level**: Rust beginner, wants to learn graphics concepts (shaders, render pipelines, GPU) as they go
- **Core technique**: LIC only (no streamlines, glyphs, particles)
- **Vector fields**: Start with procedural Perlin noise, then code-defined vortex arrays, eventually mouse-draggable vortices
- **Interactions**: Start with code tinkering, add pan/zoom, then parameter controls, then mouse interaction
- **Learning style**: Learn as you go - explain each graphics concept when introduced
- **Increments**: Very small steps, each runnable and verifiable
- **Skills to consult**: `grilling` + `domain-modeling` for each ticket; `prototype` for shader/UI decisions

## Decisions so far

- [Blue Window](WAYFINDER_TICKETS/01-blue-window.md): Create a minimal Bevy app that opens a blue window
- [Noise Texture](WAYFINDER_TICKETS/02-noise-texture.md): Generate and display a Perlin noise texture on a fullscreen quad
- [Vector Field Texture](WAYFINDER_TICKETS/03-vector-field-texture.md): Generate a 2D vector field (Perlin-based) and visualize it as color
- [LIC Shader - Basic](WAYFINDER_TICKETS/04-lic-shader-basic.md): Implement basic LIC fragment shader with noise + vector field inputs
- [LIC Shader - Triangular Weighting](WAYFINDER_TICKETS/05-lic-weighting.md): Add proper triangular weighting and normalization to LIC
- [Vortex Vector Field](WAYFINDER_TICKETS/06-vortex-field.md): Replace Perlin field with code-defined vortex array
- [Pan/Zoom Camera](WAYFINDER_TICKETS/07-pan-zoom.md): Add mouse pan/zoom controls for the vector field view
- [Interactive Vortex Parameters](WAYFINDER_TICKETS/08-vortex-params.md): UI/keyboard controls for vortex count, strength, position
- [Mouse-Draggable Vortices](WAYFINDER_TICKETS/09-mouse-vortices.md): Click/drag to move vortices, add/remove with keys

## Not yet specified

- Performance optimization (compute shaders, texture formats)
- Export/save functionality (frames, vector field data)
- Multiple LIC noise textures / contrast controls
- Time-animated vector fields
- 3D vector field visualization

## Out of scope

- Streamline/particle/glyph visualization methods (LIC only)
- Fluid simulation (Navier-Stokes) - separate project
- Web/WASM target (native only for now)
- Multi-window or multi-viewport