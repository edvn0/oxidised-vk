use super::entity_uuid::EntityUuid;
use legion::*;

pub trait WorldExt {
    fn push_with_uuid<T>(&mut self, components: T) -> (Entity, EntityUuid)
    where
        Option<T>: storage::IntoComponentSource;

    fn push_empty_with_uuid(&mut self) -> (Entity, EntityUuid);
}

impl WorldExt for World {
    fn push_with_uuid<T>(&mut self, components: T) -> (Entity, EntityUuid)
    where
        Option<T>: storage::IntoComponentSource,
    {
        let uuid = EntityUuid::new();
        let entity = self.push(components);
        if let Some(mut entry) = self.entry(entity) {
            entry.add_component(uuid.clone());
        }
        (entity, uuid)
    }

    fn push_empty_with_uuid(&mut self) -> (Entity, EntityUuid) {
        let uuid = EntityUuid::new();
        let entity = self.push((uuid.clone(),));
        (entity, uuid)
    }
}
