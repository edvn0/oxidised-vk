use crate::{FrustumPlanes, TransformTRS};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, DrawIndexedIndirectCommand,
    PrimaryAutoCommandBuffer,
};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::device::{DeviceOwned, Queue};
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::pipeline::layout::PushConstantRange;
use vulkano::pipeline::{PipelineBindPoint, PipelineShaderStageCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
    },
    descriptor_set::{WriteDescriptorSet, allocator::StandardDescriptorSetAllocator},
    device::Device,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, compute::ComputePipelineCreateInfo,
        layout::PipelineLayoutCreateInfo,
    },
    shader::ShaderStages,
    sync,
};

pub struct CullPass {
    pub pipeline: Arc<ComputePipeline>,
    pub set_layout: Arc<DescriptorSetLayout>,
}

pub struct PrefixPass {
    pub pipeline: Arc<ComputePipeline>,
    pub set_layout: Arc<DescriptorSetLayout>,
}

pub fn construct_culling_passes(device: &Arc<Device>) -> (CullPass, PrefixPass) {
    //
    // === VISIBILITY CULLING SHADER ===
    //
    mod cs_cull {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "assets/shaders/cull.comp",
        }
    }

    let cs_cull = cs_cull::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let vis_set_layout = {
        let mut b0 = DescriptorSetLayoutBinding::descriptor_type(
            vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
        );
        b0.stages = ShaderStages::COMPUTE;

        let mut b1 = DescriptorSetLayoutBinding::descriptor_type(
            vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
        );
        b1.stages = ShaderStages::COMPUTE;

        let mut b2 = DescriptorSetLayoutBinding::descriptor_type(
            vulkano::descriptor_set::layout::DescriptorType::UniformBuffer,
        );
        b2.stages = ShaderStages::COMPUTE;

        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: BTreeMap::from([
                    (0, b0), // transforms SSBO
                    (1, b1), // visibility SSBO
                    (2, b2), // frustum UBO ✔
                ]),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let vis_pipeline = {
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![vis_set_layout.clone()],
                push_constant_ranges: vec![],
                ..Default::default()
            },
        )
        .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(cs_cull),
                layout,
            ),
        )
        .unwrap()
    };

    let cull_pass = CullPass {
        pipeline: vis_pipeline,
        set_layout: vis_set_layout,
    };

    //
    // === PREFIX SCAN + INDIRECT WRITE ===
    //
    mod cs_prefix {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "assets/shaders/prefix.comp",
        }
    }

    let cs_prefix = cs_prefix::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();

    let prefix_set_layout = {
        let mut b0 = DescriptorSetLayoutBinding::descriptor_type(
            vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
        );
        b0.stages = ShaderStages::COMPUTE;

        let mut b1 = DescriptorSetLayoutBinding::descriptor_type(
            vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
        );
        b1.stages = ShaderStages::COMPUTE;

        let mut b2 = DescriptorSetLayoutBinding::descriptor_type(
            vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
        );
        b2.stages = ShaderStages::COMPUTE;

        DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: BTreeMap::from([
                    (0, b0), // Visibility mask
                    (1, b1), // Indirect buffer (multiple cmds)
                    (2, b2), // Mesh offsets
                ]),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let prefix_pipeline = {
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![prefix_set_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    offset: 0,
                    size: std::mem::size_of::<cs_prefix::Push>() as u32,
                }],
                ..Default::default()
            },
        )
        .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(cs_prefix),
                layout,
            ),
        )
        .unwrap()
    };

    let prefix_pass = PrefixPass {
        pipeline: prefix_pipeline,
        set_layout: prefix_set_layout,
    };

    (cull_pass, prefix_pass)
}

impl CullPass {
    pub fn execute(
        &self,
        cb_allocator: &Arc<StandardCommandBufferAllocator>,
        queue: &Arc<Queue>,
        allocator: &Arc<StandardDescriptorSetAllocator>,
        transforms: Subbuffer<[TransformTRS]>,
        visibility: Subbuffer<[u32]>,
        frustum: Subbuffer<FrustumPlanes>,
    ) -> Box<dyn GpuFuture> {
        let count = transforms.len() as u32;

        if count == 0 {
            return sync::now(queue.device().clone()).boxed();
        }

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let set = DescriptorSet::new(
            allocator.clone(),
            self.set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, transforms),
                WriteDescriptorSet::buffer(1, visibility),
                WriteDescriptorSet::buffer(2, frustum),
            ],
            [],
        )
        .unwrap();

        let groups = (count + 255) / 256;

        unsafe {
            command_buffer_builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Cull compute".to_string(),
                    color: [0.1, 0.2, 0.8, 1.0],
                    ..Default::default()
                })
                .unwrap();

            command_buffer_builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .dispatch([groups, 1, 1])
                .unwrap();
            command_buffer_builder.end_debug_utils_label().unwrap();
        }

        let command_buffer = command_buffer_builder.build().unwrap();
        sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .boxed()
    }
}

impl PrefixPass {
    pub fn execute(
        &self,
        cb_allocator: &Arc<StandardCommandBufferAllocator>,
        queue: &Arc<Queue>,
        allocator: &Arc<StandardDescriptorSetAllocator>,
        visibility: Subbuffer<[u32]>,
        indirect: Subbuffer<[DrawIndexedIndirectCommand]>,
        mesh_offsets: Subbuffer<[u32]>,
        mesh_count: u32,
    ) -> Box<dyn GpuFuture> {
        let count = visibility.len() as u32;

        if count == 0 {
            return sync::now(queue.device().clone()).boxed();
        }

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let set = DescriptorSet::new(
            allocator.clone(),
            self.set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, visibility),
                WriteDescriptorSet::buffer(1, indirect),
                WriteDescriptorSet::buffer(2, mesh_offsets),
            ],
            [],
        )
        .unwrap();

        let groups = (count + 255) / 256;

        unsafe {
            command_buffer_builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Prefix-sum post culling".to_string(),
                    color: [0.1, 0.9, 0.8, 1.0],
                    ..Default::default()
                })
                .unwrap();
            command_buffer_builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .push_constants(self.pipeline.layout().clone(), 0, mesh_count)
                .unwrap()
                .dispatch([groups, 1, 1])
                .unwrap();
            command_buffer_builder.end_debug_utils_label().unwrap();
        }

        let command_buffer = command_buffer_builder.build().unwrap();
        sync::now(queue.device().clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .boxed()
    }
}
