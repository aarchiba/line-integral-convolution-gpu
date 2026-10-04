//! Line Integral Convolution GPU - Binary entry point
//!
//! Shows the demo scene: fine-grain Perlin noise smeared along a central
//! vortex. The spec below is an ordinary user program input — replace the
//! generators with your own and schedule `update_lic_*` systems to animate.

use line_integral_convolution_gpu::{
    build_app,
    lic::{LicParams, LicSceneSpec},
    lic::vector_field::analytical,
    noise::generate_noise_grayscale,
};

fn main() {
    build_app(LicSceneSpec::new(
        640,
        360,
        |w, h, _t| generate_noise_grayscale(w, h, 0.06),
        |w, h, _t| {
            let cx = (w - 1) as f32 / 2.0;
            let cy = (h - 1) as f32 / 2.0;
            analytical::vortex(w, h, (cx, cy), 60.0)
        },
        |_t| LicParams::triangular_kernel(12),
    ))
    .run();
}
