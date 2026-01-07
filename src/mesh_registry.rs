use crate::mesh::{MaterialClass, MeshAsset};
use std::collections::HashMap;
use std::sync::Arc;

#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MeshHandle(pub u32);

#[derive(Hash, Eq, PartialEq, Copy, Clone, Debug)]
pub struct RenderStreamKey {
    pub mesh: MeshHandle,
    pub material_class: MaterialClass,
}

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

    pub fn resolve_and_get(&self, name: &str) -> Option<(&Arc<MeshAsset>, MeshHandle)> {
        let handle = self.resolve(name);
        handle.and_then(|handle| self.get(handle).ok().map(|mesh| (mesh, handle)))
    }

    pub fn get(&self, handle: MeshHandle) -> Result<&Arc<MeshAsset>, String> {
        self.meshes
            .get(handle.0 as usize)
            .ok_or_else(|| format!("Mesh handle {} is invalid", handle.0))
    }

    pub fn iter(&self) -> impl Iterator<Item = (MeshHandle, &Arc<MeshAsset>)> {
        self.meshes
            .iter()
            .enumerate()
            .map(|(i, mesh)| (MeshHandle(i as u32), mesh))
    }
}
