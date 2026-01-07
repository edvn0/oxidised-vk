use glm::{Quat, Vec3};
use nalgebra::{Matrix4, Vector3, Vector4};

use crate::aabb::Aabb;
use crate::mesh_registry::MeshHandle;
use crate::submission::SubmeshSelection;

#[derive(Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Quat,
}

impl Transform {
    pub fn to_matrix(&self) -> [f32; 16] {
        let t = glm::translate(&glm::identity(), &self.position);
        let r = glm::quat_to_mat4(&self.rotation.normalize());
        let s = glm::scale(&glm::identity(), &self.scale);

        let mat = t * r * s;

        mat.as_slice().try_into().unwrap()
    }

    fn to_internal_matrix(&self) -> Matrix4<f32> {
        let t = glm::translate(&glm::identity(), &self.position);
        let r = glm::quat_to_mat4(&self.rotation.normalize());
        let s = glm::scale(&glm::identity(), &self.scale);

        t * r * s
    }

    pub fn apply_point(&self, p: Vector3<f32>) -> Vector3<f32> {
        let m = self.to_internal_matrix();
        let v = m * Vector4::new(p.x, p.y, p.z, 1.0);
        Vector3::new(v.x, v.y, v.z)
    }
}

pub struct MeshComponent {
    pub mesh: MeshHandle,
    pub submeshes: SubmeshSelection,
}
pub struct Bounds {
    pub submeshes: Vec<Aabb>,
}

/*
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
*/

pub struct MaterialOverride {
    pub material_id: u32,
}

pub struct Visible;
