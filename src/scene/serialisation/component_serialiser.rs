use std::io::{Read, Result as IoResult, Write};

use legion::world::EntryRef;

pub trait SerializableComponent: Send + Sync + 'static {
    fn serialize(&self, writer: &mut dyn Write) -> IoResult<()>;
    fn deserialize(reader: &mut dyn Read) -> IoResult<Self>
    where
        Self: Sized;
    fn type_name() -> &'static str
    where
        Self: Sized;
}

pub trait ComponentSerialiser: Send + Sync {
    fn serialize_component(&self, entry: &EntryRef<'_>, writer: &mut dyn Write) -> IoResult<()>;

    fn deserialize_component(
        &self,
        reader: &mut dyn Read,
        entry: &mut legion::world::Entry<'_>,
    ) -> IoResult<()>;
}

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
    use glm::{Quat, Vec3};

    use crate::{
        components::{self, MeshComponent, Transform},
        mesh_registry::MeshHandle,
        scene::serialisation::{SerializableComponent, deserialize_pod, serialize_pod},
    };
    use std::io::{Read, Write};

    impl SerializableComponent for Transform {
        fn serialize(&self, writer: &mut dyn Write) -> std::io::Result<()> {
            // Convert to arrays and serialize
            let position: [f32; 3] = [self.position.x, self.position.y, self.position.z];
            writer.write_all(bytemuck::bytes_of(&position))?;

            let scale: [f32; 3] = [self.scale.x, self.scale.y, self.scale.z];
            writer.write_all(bytemuck::bytes_of(&scale))?;

            let rotation: [f32; 4] = [
                self.rotation.w,
                self.rotation.i,
                self.rotation.j,
                self.rotation.k,
            ];
            writer.write_all(bytemuck::bytes_of(&rotation))?;

            Ok(())
        }

        fn deserialize(reader: &mut dyn Read) -> std::io::Result<Self> {
            // Deserialize position
            let mut position: [f32; 3] = [0.0; 3];
            reader.read_exact(bytemuck::bytes_of_mut(&mut position))?;

            // Deserialize scale
            let mut scale: [f32; 3] = [0.0; 3];
            reader.read_exact(bytemuck::bytes_of_mut(&mut scale))?;

            // Deserialize rotation
            let mut rotation: [f32; 4] = [0.0; 4];
            reader.read_exact(bytemuck::bytes_of_mut(&mut rotation))?;

            Ok(Self {
                position: Vec3::new(position[0], position[1], position[2]),
                scale: Vec3::new(scale[0], scale[1], scale[2]),
                rotation: Quat::new(rotation[0], rotation[1], rotation[2], rotation[3]),
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

#[cfg(test)]
mod tests {
    use glm::{Quat, Vec3};
    use nalgebra::{Unit, UnitQuaternion};

    use crate::components::Transform;

    use super::*;
    use std::io::Cursor;

    #[test]
    fn transform_identity_roundtrip() {
        let transform = Transform {
            position: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(1.0, 1.0, 1.0),
            rotation: Quat::identity(),
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let deserialized = Transform::deserialize(&mut cursor).unwrap();

        assert_eq!(transform.position, deserialized.position);
        assert_eq!(transform.scale, deserialized.scale);
        assert_eq!(transform.rotation, deserialized.rotation);
    }

    #[test]
    fn transform_translated_roundtrip() {
        let transform = Transform {
            position: Vec3::new(10.5, -5.2, 3.14),
            scale: Vec3::new(1.0, 1.0, 1.0),
            rotation: Quat::identity(),
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let deserialized = Transform::deserialize(&mut cursor).unwrap();

        assert_eq!(transform.position, deserialized.position);
        assert_eq!(transform.scale, deserialized.scale);
        assert_eq!(transform.rotation, deserialized.rotation);
    }

    #[test]
    fn transform_scaled_roundtrip() {
        let transform = Transform {
            position: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(2.0, 0.5, 3.0),
            rotation: Quat::identity(),
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let deserialized = Transform::deserialize(&mut cursor).unwrap();

        assert_eq!(transform.position, deserialized.position);
        assert_eq!(transform.scale, deserialized.scale);
        assert_eq!(transform.rotation, deserialized.rotation);
    }

    #[test]
    fn transform_rotated_roundtrip() {
        // 90 degree rotation around Y axis using from_axis_angle
        let rotation =
            UnitQuaternion::from_axis_angle(&Vec3::y_axis(), std::f32::consts::FRAC_PI_2);

        let transform = Transform {
            position: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(1.0, 1.0, 1.0),
            rotation: rotation.into_inner(),
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let deserialized = Transform::deserialize(&mut cursor).unwrap();

        assert_eq!(transform.position, deserialized.position);
        assert_eq!(transform.scale, deserialized.scale);

        // Compare quaternion components with small epsilon for floating point errors
        let epsilon = 1e-6;
        assert!((transform.rotation.coords[0] - deserialized.rotation.coords[0]).abs() < epsilon);
        assert!((transform.rotation.coords[1] - deserialized.rotation.coords[1]).abs() < epsilon);
        assert!((transform.rotation.coords[2] - deserialized.rotation.coords[2]).abs() < epsilon);
        assert!((transform.rotation.coords[3] - deserialized.rotation.coords[3]).abs() < epsilon);
    }

    #[test]
    fn transform_full_roundtrip() {
        // Create rotation using from_axis_angle and a custom axis
        let axis = Unit::new_normalize(Vec3::new(1.0, 2.0, 3.0));
        let rotation = UnitQuaternion::from_axis_angle(&axis, 1.5);

        let transform = Transform {
            position: Vec3::new(100.0, -50.0, 25.5),
            scale: Vec3::new(2.5, 1.5, 0.8),
            rotation: rotation.into_inner(),
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let deserialized = Transform::deserialize(&mut cursor).unwrap();

        assert_eq!(transform.position, deserialized.position);
        assert_eq!(transform.scale, deserialized.scale);

        let epsilon = 1e-6;
        assert!((transform.rotation.coords[0] - deserialized.rotation.coords[0]).abs() < epsilon);
        assert!((transform.rotation.coords[1] - deserialized.rotation.coords[1]).abs() < epsilon);
        assert!((transform.rotation.coords[2] - deserialized.rotation.coords[2]).abs() < epsilon);
        assert!((transform.rotation.coords[3] - deserialized.rotation.coords[3]).abs() < epsilon);
    }

    #[test]
    fn transform_negative_scale_roundtrip() {
        let transform = Transform {
            position: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(-1.0, 1.0, -1.0),
            rotation: Quat::identity(),
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let deserialized = Transform::deserialize(&mut cursor).unwrap();

        assert_eq!(transform.position, deserialized.position);
        assert_eq!(transform.scale, deserialized.scale);
        assert_eq!(transform.rotation, deserialized.rotation);
    }

    #[test]
    fn transform_serialized_size() {
        let transform = Transform {
            position: Vec3::new(1.0, 2.0, 3.0),
            scale: Vec3::new(1.0, 1.0, 1.0),
            rotation: Quat::identity(),
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        // Should be 3 floats (position) + 3 floats (scale) + 4 floats (rotation) = 10 * 4 bytes
        assert_eq!(buffer.len(), 40);
    }

    #[test]
    fn transform_multiple_rotations_roundtrip() {
        // Create a rotation by composing multiple axis-angle rotations
        let rot_x = UnitQuaternion::from_axis_angle(&Vec3::x_axis(), 0.5);
        let rot_y = UnitQuaternion::from_axis_angle(&Vec3::y_axis(), 1.2);
        let rotation = (rot_y * rot_x).into_inner();

        let transform = Transform {
            position: Vec3::new(5.0, 10.0, 15.0),
            scale: Vec3::new(1.0, 2.0, 3.0),
            rotation,
        };

        let mut buffer = Vec::new();
        transform.serialize(&mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let deserialized = Transform::deserialize(&mut cursor).unwrap();

        assert_eq!(transform.position, deserialized.position);
        assert_eq!(transform.scale, deserialized.scale);

        let epsilon = 1e-6;
        assert!((transform.rotation.coords[0] - deserialized.rotation.coords[0]).abs() < epsilon);
        assert!((transform.rotation.coords[1] - deserialized.rotation.coords[1]).abs() < epsilon);
        assert!((transform.rotation.coords[2] - deserialized.rotation.coords[2]).abs() < epsilon);
        assert!((transform.rotation.coords[3] - deserialized.rotation.coords[3]).abs() < epsilon);
    }
}
