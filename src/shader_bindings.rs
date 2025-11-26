use vulkano::buffer::BufferContents;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::shader::ShaderStages;

#[derive(BufferContents)]
#[repr(C)]
pub struct RendererUBO {
    pub view: [f32; 16],
    pub proj: [f32; 16],
    pub inverse_proj: [f32; 16],
    pub sun_direction: [f32; 4],
}

pub fn renderer_set_0_layouts() -> Vec<DescriptorSetLayoutCreateInfo> {
    let ubo_layout = {
        let mut frame_camera_renderer_ubo =
            DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);
        frame_camera_renderer_ubo.stages = ShaderStages::all_graphics();

        let mut frame_transform_buffer =
            DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
        frame_transform_buffer.stages = ShaderStages::all_graphics();

        let mut info = DescriptorSetLayoutCreateInfo::default();
        info.bindings.insert(0, frame_camera_renderer_ubo);
        info.bindings.insert(1, frame_transform_buffer);
        info
    };

    vec![ubo_layout]
}
