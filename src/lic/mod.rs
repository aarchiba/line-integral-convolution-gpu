//! Line Integral Convolution (LIC) module

pub mod cpu_reference;
pub mod params;
pub mod material;
pub mod scene;
pub mod vector_field;
pub mod pipeline;

pub use pipeline::{LicPlugin, OffscreenTargets, create_offscreen_targets};
pub use material::LicMaterial;
pub use scene::{
    FieldGenerator, KernelGenerator, LicScene, LicSceneSpec, NoiseGenerator,
    setup_lic_scene, update_lic_field, update_lic_ink, update_lic_kernel,
};
pub use params::LicParams;
pub use vector_field::{VectorFieldNode, CpuVectorFieldNode, PrecookedVectorFieldNode};