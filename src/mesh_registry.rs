use std::collections::HashMap;
use std::sync::Arc;
use crate::mesh::MeshAsset;

pub struct MeshRegistry {
    meshes: HashMap<String, Arc<MeshAsset>>,
}

impl MeshRegistry {
    pub fn new(meshes: HashMap<String, Arc<MeshAsset>>) -> Self {
        Self { meshes }
    }

    pub fn get(&self, name: &str) -> Option<Arc<MeshAsset>> {
        self.meshes.get(name).cloned()
    }

    pub fn insert(&mut self, name: String, mesh: Arc<MeshAsset>) {
        self.meshes.insert(name, mesh);
    }

    pub fn remove(&mut self, name: &str) -> Option<Arc<MeshAsset>> {
        self.meshes.remove(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<MeshAsset>)> {
        self.meshes.iter()
    }
}