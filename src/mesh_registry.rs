use std::collections::HashMap;
use std::sync::Arc;
use crate::mesh::MeshAsset;

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MeshHandle(pub u32);

pub struct MeshRegistry {
    meshes: Vec<Arc<MeshAsset>>,
    name_to_handle: HashMap<String, MeshHandle>,
}

impl MeshRegistry {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            name_to_handle: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, mesh: Arc<MeshAsset>) -> MeshHandle {
        if let Some(handle) = self.name_to_handle.get(&name) {
            return *handle;
        }

        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(mesh);
        self.name_to_handle.insert(name, handle);
        handle
    }

    pub fn resolve(&self, name: &str) -> Option<MeshHandle> {
        self.name_to_handle.get(name).copied()
    }

    pub fn get(&self, handle: MeshHandle) -> &Arc<MeshAsset> {
        &self.meshes[handle.0 as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = (MeshHandle, &Arc<MeshAsset>)> {
        self.meshes
            .iter()
            .enumerate()
            .map(|(i, mesh)| (MeshHandle(i as u32), mesh))
    }
}
