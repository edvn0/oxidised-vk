use std::sync::Arc;
use crate::mesh::MeshAsset;
use crate::TransformTRS;

#[derive(Clone, Copy)]
pub struct Transform {
    pub trs: TransformTRS,
}

pub struct MeshComponent {
    pub mesh: Arc<MeshAsset>,
}

pub struct MaterialOverride {
    pub material_id: u32,
}

pub struct Visible;