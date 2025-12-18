use crate::TransformTRS;
use crate::mesh_registry::MeshHandle;

pub struct DrawSubmission {
    pub mesh: MeshHandle,
    pub transform: TransformTRS,
    pub override_material: Option<u32>,
}

pub struct FrameSubmission {
    pub draws: Vec<DrawSubmission>,
}

impl FrameSubmission {
    pub fn new() -> Self {
        Self { draws: Vec::new() }
    }

    pub fn clear_all(&mut self) {
        self.draws.clear();
    }

    pub fn drain_draws_into(&mut self, target: &mut Vec<DrawSubmission>) {
        target.extend(self.draws.drain(..));
    }
}

impl Default for FrameSubmission {
    fn default() -> Self {
        Self::new()
    }
}
