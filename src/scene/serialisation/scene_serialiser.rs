use legion::*;
use std::io::{Read, Result as IoResult, Write};

use super::component_registry::ComponentRegistry;
use super::component_serialiser::SerializableComponent;
use crate::scene::WorldExt;
use crate::scene::entity_uuid::EntityUuid;

pub struct SceneSerialiser {
    registry: ComponentRegistry,
}

impl SceneSerialiser {
    pub fn new() -> Self {
        let mut registry = ComponentRegistry::new();
        registry.register::<EntityUuid>();

        Self { registry }
    }

    pub fn register<T: SerializableComponent>(&mut self) -> &mut Self {
        self.registry.register::<T>();
        self
    }

    pub fn serialize_world(&self, world: &World, writer: &mut dyn Write) -> IoResult<()> {
        writer.write_all(b"LGSN")?;
        writer.write_all(&1u32.to_le_bytes())?;

        let entity_count = world.len() as u64;
        writer.write_all(&entity_count.to_le_bytes())?;

        let mut query = <(Entity, &EntityUuid)>::query();
        for (entity, _) in query.iter(world) {
            self.serialize_entity(world, *entity, writer)?;
        }

        Ok(())
    }

    fn serialize_entity(
        &self,
        world: &World,
        entity: Entity,
        writer: &mut dyn Write,
    ) -> IoResult<()> {
        let entry = world.entry_ref(entity).unwrap();

        let mut mask: u64 = 0;

        let archetype = entry.archetype();
        let layout = archetype.layout();

        for ct in layout.component_types() {
            if let Some(bit) = self.registry.get_bit(&ct.type_id()) {
                mask |= 1u64 << bit.0;
            }
        }

        let uuid_bit = self
            .registry
            .get_bit(&std::any::TypeId::of::<EntityUuid>())
            .unwrap();

        if (mask & (1u64 << uuid_bit.0)) == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Entity missing EntityUuid",
            ));
        }

        writer.write_all(&mask.to_le_bytes())?;

        for bit in 0..self.registry.component_count() {
            if (mask & (1u64 << bit)) == 0 {
                continue;
            }

            let serialiser = self.registry.serialiser_for_bit(bit as u8);
            serialiser.serialize_component(&entry, writer)?;
        }

        Ok(())
    }

    pub fn deserialize_world(&self, reader: &mut dyn Read) -> IoResult<World> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"LGSN" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic number",
            ));
        }

        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported version {}", version),
            ));
        }

        let mut count_bytes = [0u8; 8];
        reader.read_exact(&mut count_bytes)?;
        let entity_count = u64::from_le_bytes(count_bytes);

        let mut world = World::default();

        for _ in 0..entity_count {
            self.deserialize_entity(&mut world, reader)?;
        }

        Ok(world)
    }

    fn deserialize_entity(&self, world: &mut World, reader: &mut dyn Read) -> IoResult<()> {
        let mut mask_bytes = [0u8; 8];
        reader.read_exact(&mut mask_bytes)?;
        let mask = u64::from_le_bytes(mask_bytes);

        let (entity, _) = world.push_empty_with_uuid();
        let mut entry = world.entry(entity).unwrap();

        let uuid_bit = self
            .registry
            .get_bit(&std::any::TypeId::of::<EntityUuid>())
            .unwrap();

        if (mask & (1u64 << uuid_bit.0)) == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Entity missing EntityUuid",
            ));
        }

        for bit in 0..self.registry.component_count() {
            if (mask & (1u64 << bit)) == 0 {
                continue;
            }

            let serialiser = self.registry.serialiser_for_bit(bit as u8);
            serialiser.deserialize_component(reader, &mut entry)?;
        }

        Ok(())
    }
}
