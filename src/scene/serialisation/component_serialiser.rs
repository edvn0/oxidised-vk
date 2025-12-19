use std::any::TypeId;
use std::io::{Read, Result as IoResult, Write};

use legion::world::EntryRef;

/// Trait for components that can be serialized to/from binary format
pub trait SerializableComponent: Send + Sync + 'static {
    fn serialize(&self, writer: &mut dyn Write) -> IoResult<()>;
    fn deserialize(reader: &mut dyn Read) -> IoResult<Self>
    where
        Self: Sized;
    fn type_id() -> TypeId
    where
        Self: Sized,
    {
        TypeId::of::<Self>()
    }
    fn type_name() -> &'static str
    where
        Self: Sized;
}

/// Trait for type-erased component serialisation
pub trait ComponentSerialiser: Send + Sync {
    fn serialize_component(&self, entry: &EntryRef<'_>, writer: &mut dyn Write) -> IoResult<()>;

    fn deserialize_component(
        &self,
        reader: &mut dyn Read,
        entry: &mut legion::world::Entry<'_>,
    ) -> IoResult<()>;
}

/// Typed implementation of ComponentSerialiser
pub(crate) struct TypedComponentSerialiser<T: SerializableComponent> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T: SerializableComponent> TypedComponentSerialiser<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: SerializableComponent + legion::storage::Component> ComponentSerialiser
    for TypedComponentSerialiser<T>
{
    fn serialize_component(&self, entry: &EntryRef<'_>, writer: &mut dyn Write) -> IoResult<()> {
        if let Ok(component) = entry.get_component::<T>() {
            component.serialize(writer)
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Component not found on entity",
            ))
        }
    }

    fn deserialize_component(
        &self,
        reader: &mut dyn Read,
        entry: &mut legion::world::Entry<'_>,
    ) -> IoResult<()> {
        let component = T::deserialize(reader)?;
        entry.add_component(component);
        Ok(())
    }
}

mod serialiser_implementations {
    use crate::{
        TransformTRS,
        components::{self, MeshComponent, Transform},
        mesh_registry::MeshHandle,
        scene::serialisation::{SerializableComponent, deserialize_pod, serialize_pod},
    };
    use std::io::{Read, Write};

    type TransformSlice = [f32; 16];
    const BYTES_PER_TRANSFORM: usize = std::mem::size_of::<TransformSlice>();

    impl SerializableComponent for Transform {
        fn serialize(&self, writer: &mut dyn Write) -> std::io::Result<()> {
            let bytes: &[u8] = bytemuck::try_cast_slice::<f32, u8>(&self.transform.trs)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
            writer.write_all(bytes)
        }

        fn deserialize(reader: &mut dyn Read) -> std::io::Result<Self> {
            let mut bytes = [0u8; BYTES_PER_TRANSFORM];
            reader.read_exact(&mut bytes)?;

            let trs: TransformSlice = *bytemuck::try_from_bytes::<TransformSlice>(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

            Ok(Self {
                transform: TransformTRS { trs },
            })
        }

        fn type_name() -> &'static str {
            "Transform"
        }
    }

    impl SerializableComponent for MeshComponent {
        fn serialize(&self, writer: &mut dyn Write) -> std::io::Result<()> {
            serialize_pod::<u32>(&self.mesh.0, writer)
        }

        fn deserialize(reader: &mut dyn Read) -> std::io::Result<Self> {
            let mesh_id = deserialize_pod::<u32>(reader)?;
            Ok(Self {
                mesh: MeshHandle(mesh_id),
            })
        }

        fn type_name() -> &'static str {
            "MeshComponent"
        }
    }

    impl SerializableComponent for components::Visible {
        fn serialize(&self, _writer: &mut dyn Write) -> std::io::Result<()> {
            Ok(())
        }

        fn deserialize(_reader: &mut dyn Read) -> std::io::Result<Self> {
            Ok(components::Visible)
        }

        fn type_name() -> &'static str {
            "Visible"
        }
    }

    impl SerializableComponent for components::MaterialOverride {
        fn serialize(&self, writer: &mut dyn Write) -> std::io::Result<()> {
            serialize_pod(&self.material_id, writer)
        }

        fn deserialize(reader: &mut dyn Read) -> std::io::Result<Self> {
            deserialize_pod(reader).map(|material_id: u32| Self { material_id })
        }

        fn type_name() -> &'static str {
            "MaterialOverride"
        }
    }
}
