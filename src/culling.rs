use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::pipeline::layout::PushConstantRange;
use vulkano::pipeline::PipelineShaderStageCreateInfo;
use vulkano::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
    }
    ,
    device::Device
    ,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineLayoutCreateInfo, ComputePipeline,
        PipelineLayout,
    },
    shader::ShaderStages
    ,
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
