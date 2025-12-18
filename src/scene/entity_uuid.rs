use super::serialisation::SerializableComponent;
use std::io::{Read, Result as IoResult, Write};
use uuid::Uuid;

/// Component that uniquely identifies an entity across serialisation
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct EntityUuid(pub Uuid);

impl EntityUuid {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl SerializableComponent for EntityUuid {
    fn serialize(&self, writer: &mut dyn Write) -> IoResult<()> {
        writer.write_all(self.0.as_bytes())
    }

    fn deserialize(reader: &mut dyn Read) -> IoResult<Self> {
        let mut bytes = [0u8; 16];
        reader.read_exact(&mut bytes)?;
        Ok(Self(Uuid::from_bytes(bytes)))
    }

    fn type_name() -> &'static str {
        "EntityUuid"
    }
}
