# Sampler-native boundary modes, no Zero

Boundary handling on the LIC module is sampler-native only: per-axis `ClampToEdge` (default), `Repeat`, or `MirrorRepeat` via Bevy `ImageSamplerDescriptor`, applied identically to noise and vector-field sampling. There is no `Zero`/`Constant` mode because sampler-level zero (`ClampToBorder`) needs the opt-in wgpu `ADDRESS_MODE_CLAMP_TO_BORDER` feature, and a manual shader branch would add a second code path that the CPU reference must also mirror.

## Considered Options

- `Zero` via manual `pos in [0,1]` bounds-check in `lic.wgsl` + CPU mirror (keep-weight normalization, `BORDER_CONSTANT` semantics). Rejected: extra per-tap branch for a use case nobody has yet; easy to add later.
- Custom `Zero / Clamp / Wrap` enum aliases. Rejected: a translation layer over Bevy's `ImageAddressMode` with no benefit; OpenCV mapping (`Replicate→ClampToEdge`, `Wrap→Repeat`) lives in docs instead.
- Per-edge modes. Rejected: per-edge wrap is incoherent (not a tiling); per-axis covers the real mix case (e.g. periodic in x, clamped in y).

## Consequences

- `wsum` stays kernel-constant (every tap resolves in-bounds), so kernel scale-invariance holds trivially and the existing `acc / wsum` contract is untouched.
- If an edge-darkening use case appears, add `Zero` as a uniform-driven shader branch plus CPU parity; this ADR then gets superseded.
