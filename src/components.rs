use crate::TransformTRS;
use crate::mesh_registry::{MeshHandle, MeshRegistry};

#[derive(Clone, Copy)]
pub struct Transform {
    pub transform: TransformTRS,
}

pub struct MeshComponent {
    pub mesh: MeshHandle,
}
pub struct MeshComponentSerialized {
    pub mesh_name: String,
}
impl MeshComponent {
    pub fn from_serialized(s: MeshComponentSerialized, registry: &MeshRegistry) -> Self {
        let handle = registry
            .resolve(&s.mesh_name)
            .expect("mesh not found in registry");

        Self { mesh: handle }
    }
}

pub struct MaterialOverride {
    pub material_id: u32,
}

pub struct Visible;
