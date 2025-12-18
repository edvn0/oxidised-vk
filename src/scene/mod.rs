pub mod entity_uuid;
pub mod serialisation;
pub mod world_ext;

use crate::components::{MaterialOverride, MeshComponent, Transform, Visible};
use crate::scene::serialisation::scene_serialiser::SceneSerialiser;
use legion::*;
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

    pub fn create_serialiser() -> SceneSerialiser {
        let mut serialiser = SceneSerialiser::new();
        serialiser
            .register::<Transform>()
            .register::<MeshComponent>()
            .register::<Visible>()
            .register::<MaterialOverride>();
        serialiser
    }

    pub fn save_to_file(&self, path: &Path) -> IoResult<()> {
        let serialiser = Self::create_serialiser();
        let mut file = std::fs::File::create(path)?;
        serialiser.serialize_world(&self.world, &mut file)
    }

    pub fn load_from_file(path: &Path) -> IoResult<Self> {
        let serialiser = Self::create_serialiser();
        let mut file = std::fs::File::open(path)?;
        let world = serialiser.deserialize_world(&mut file)?;

        Ok(Self::from_world(world))
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
        transform: transform.transform,
        override_material: material.map(|m| m.material_id),
    });
}
