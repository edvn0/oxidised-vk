use legion::*;
use crate::components::{MaterialOverride, MeshComponent, Transform, Visible};
use crate::{DrawSubmission, FrameSubmission};

pub struct Scene {
    pub world: World,
    pub resources: Resources,
    pub schedule: Schedule,
}

impl Scene {
    pub fn new() -> Self {
        let schedule = Schedule::builder()
            .add_system(collect_draws_system())
            .build();

        Self {
            world: World::default(),
            resources: Resources::default(),
            schedule,
        }
    }

    pub fn update(&mut self) {
        self.schedule.execute(&mut self.world, &mut self.resources);
    }
}

#[system(for_each)]
fn collect_draws(
    transform: &Transform,
    mesh: &MeshComponent,
    material: Option<&MaterialOverride>,
    _: &Visible,
    #[resource] submission: &mut FrameSubmission,
) {
    submission.draws.push(DrawSubmission {
        mesh: mesh.mesh.clone(),
        transform: transform.trs,
        override_material: material.map(|m| m.material_id),
    });
}
