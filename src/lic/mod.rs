//! Line Integral Convolution (LIC) module

pub mod params;
pub mod material;
pub mod vector_field;
pub mod pipeline;

pub use pipeline::LicPlugin;
pub use params::LicParams;
pub use vector_field::{VectorFieldNode, CpuVectorFieldNode, PrecookedVectorFieldNode};