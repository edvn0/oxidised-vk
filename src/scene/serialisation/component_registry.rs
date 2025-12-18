use std::any::TypeId;
use std::collections::HashMap;

use super::component_serialiser::{ComponentSerialiser, SerializableComponent};
use crate::scene::serialisation::component_serialiser::TypedComponentSerialiser;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct ComponentBit(pub u8);

pub struct ComponentRegistry {
    type_to_bit: HashMap<TypeId, ComponentBit>,
    bit_to_type: Vec<TypeId>,
    serialisers: Vec<Box<dyn ComponentSerialiser>>,

    type_names: HashMap<TypeId, &'static str>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        Self {
            type_to_bit: HashMap::new(),
            bit_to_type: Vec::new(),
            serialisers: Vec::new(),
            type_names: HashMap::new(),
        }
    }

    pub fn register<T: SerializableComponent>(&mut self) {
        let type_id = TypeId::of::<T>();

        if self.type_to_bit.contains_key(&type_id) {
            return;
        }

        let bit = ComponentBit(self.bit_to_type.len() as u8);
        assert!(bit.0 < 64, "Exceeded 64 component limit");

        self.type_to_bit.insert(type_id, bit);
        self.bit_to_type.push(type_id);
        self.serialisers
            .push(Box::new(TypedComponentSerialiser::<T>::new()));

        self.type_names.insert(type_id, T::type_name());
    }

    pub fn component_count(&self) -> usize {
        self.bit_to_type.len()
    }

    pub fn get_bit(&self, type_id: &TypeId) -> Option<ComponentBit> {
        self.type_to_bit.get(type_id).copied()
    }

    pub fn serialiser_for_bit(&self, bit: u8) -> &dyn ComponentSerialiser {
        self.serialisers[bit as usize].as_ref()
    }
}
