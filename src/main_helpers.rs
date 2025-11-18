use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::device::Device;
use vulkano::shader::ShaderStages;
use crate::shader_bindings::{renderer_set_0_layouts, RendererUBO};

pub struct FrameDescriptorSet {
    sets: Vec<Arc<DescriptorSet>>,
}

impl FrameDescriptorSet {
    pub fn new(
        device: Arc<Device>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        uniform_buffers: &[Subbuffer<RendererUBO>],
    ) -> Self {
        let set_0 = renderer_set_0_layouts();
        let layout = DescriptorSetLayout::new(device.clone(), set_0[0usize].clone()).unwrap();

        let sets: Vec<_> = uniform_buffers
            .iter()
            .map(|ub| {
                let mut writes = Vec::new();
                writes.push(WriteDescriptorSet::buffer(0, ub.clone()));
                DescriptorSet::new(descriptor_set_allocator.clone(), layout.clone(), writes, [])
                    .unwrap()
            })
            .collect();

        Self { sets }
    }

    pub fn for_frame(&self, index: usize) -> &Arc<DescriptorSet> {
        &self.sets[index]
    }
}