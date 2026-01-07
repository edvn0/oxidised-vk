pub mod entity_uuid;
pub mod panel;
pub mod serialisation;
pub mod world_ext;

use crate::camera::Ray;
use crate::components::{Bounds, MaterialOverride, MeshComponent, Transform, Visible};
use crate::scene::serialisation::scene_serialiser::SceneSerialiser;
use legion::*;
use nalgebra::Vector3;
use std::io::Result as IoResult;
use std::path::Path;

use crate::scene::entity_uuid::EntityUuid;
use crate::submission::{DrawSubmission, FrameSubmission};
pub use world_ext::WorldExt;

pub struct Scene {
    world: World,
    resources: Resources,
    schedule: Schedule,
}

impl Scene {
    pub fn new() -> Self {
        Self::from_world(World::default())
    }

    pub fn from_world(world: World) -> Self {
        let schedule = Schedule::builder()
            .add_system(collect_draws_system())
            .flush()
            .build();

        let mut resources = Resources::default();
        resources.insert(FrameSubmission { draws: vec![] });

        Self {
            world,
            resources,
            schedule,
        }
    }

    pub fn add_entity<T>(&mut self, components: T) -> (Entity, EntityUuid)
    where
        Option<T>: storage::IntoComponentSource,
    {
        self.world.push_with_uuid(components)
    }

    pub fn resources_mut(&mut self) -> &mut Resources {
        &mut self.resources
    }

    pub fn update(&mut self) {
        self.schedule.execute(&mut self.world, &mut self.resources);
    }

    /// Create a serialiser with all registered component types
    pub fn create_serialiser() -> SceneSerialiser {
        let mut serialiser = SceneSerialiser::new();
        // EntityUuid is already registered by default
        serialiser
            .register::<Transform>()
            .register::<MeshComponent>()
            .register::<Visible>()
            .register::<MaterialOverride>();
        serialiser
    }

    /// Save the scene to a file
    pub fn save_to_file(&self, path: &Path) -> IoResult<()> {
        let serialiser = Self::create_serialiser();
        let mut file = std::fs::File::create(path)?;
        serialiser.serialize_world(&self.world, &mut file)
    }

    /// Load a scene from a file
    pub fn load_from_file(path: &Path) -> IoResult<Self> {
        let serialiser = Self::create_serialiser();
        let mut file = std::fs::File::open(path)?;
        let world = serialiser.deserialize_world(&mut file)?;

        Ok(Self::from_world(world))
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    fn world(&mut self) -> &World {
        &self.world
    }

    pub fn pick(&self, ray: &Ray) -> Option<EntityUuid> {
        let mut closest = None;
        let mut best_t = f32::INFINITY;

        let mut query = <(&Transform, &Visible, &Bounds, &EntityUuid)>::query();

        for (transform, _visible, bounds, uuid) in query.iter(&self.world) {
            for aabb in &bounds.submeshes {
                let world_min = transform.apply_point(aabb.min);
                let world_max = transform.apply_point(aabb.max);

                if let Some(t) = ray_aabb(ray, world_min, world_max) {
                    if t < best_t {
                        best_t = t;
                        closest = Some(*uuid);
                    }
                }
            }
        }

        closest
    }
}

#[system(for_each)]
fn collect_draws(
    _uuid: Option<&EntityUuid>,
    transform: &Transform,
    mesh: &MeshComponent,
    material: Option<&MaterialOverride>,
    _: &Visible,
    #[resource] submission: &mut FrameSubmission,
) {
    submission.draws.push(DrawSubmission {
        mesh: mesh.mesh,
        transform: transform.to_matrix(),
        override_material: material.map(|m| m.material_id),
        submesh: mesh.submeshes.clone(),
    });
}

pub fn ray_aabb(ray: &Ray, min: Vector3<f32>, max: Vector3<f32>) -> Option<f32> {
    let inv_dir = Vector3::new(
        1.0 / ray.direction.x,
        1.0 / ray.direction.y,
        1.0 / ray.direction.z,
    );

    let mut t1 = (min.x - ray.origin.x) * inv_dir.x;
    let mut t2 = (max.x - ray.origin.x) * inv_dir.x;
    let mut tmin = t1.min(t2);
    let mut tmax = t1.max(t2);

    t1 = (min.y - ray.origin.y) * inv_dir.y;
    t2 = (max.y - ray.origin.y) * inv_dir.y;
    tmin = tmin.max(t1.min(t2));
    tmax = tmax.min(t1.max(t2));

    t1 = (min.z - ray.origin.z) * inv_dir.z;
    t2 = (max.z - ray.origin.z) * inv_dir.z;
    tmin = tmin.max(t1.min(t2));
    tmax = tmax.min(t1.max(t2));

    if tmax >= tmin.max(0.0) {
        Some(tmin.max(0.0))
    } else {
        None
    }
}
