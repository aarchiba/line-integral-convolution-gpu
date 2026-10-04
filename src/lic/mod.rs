//! Line Integral Convolution (LIC) module

pub mod cpu_reference;
pub mod params;
pub mod material;
pub mod vector_field;
pub mod pipeline;

pub use pipeline::{LicPlugin, OffscreenTargets, create_offscreen_targets};
pub use material::{LicMaterial, setup_lic_display};
pub use params::LicParams;
pub use vector_field::{VectorFieldNode, CpuVectorFieldNode, PrecookedVectorFieldNode};