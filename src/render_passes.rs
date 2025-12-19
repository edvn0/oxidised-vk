use std::{collections::HashMap, sync::Arc};

use vulkano::{
    Validated, ValidationError, VulkanError,
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    format::ClearValue,
    image::view::ImageView,
    instance::debug::DebugUtilsLabel,
    pipeline::{
        graphics::{depth_stencil::CompareOp, viewport::Scissor},
        *,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
};

use crate::{
    mesh::ImageViewSampler,
    mesh_registry::MeshHandle,
    render_context::{Culling, MeshDrawStream, RenderContext, Winding},
};

#[derive(Debug)]
pub enum RenderPassError {
    Vulkan(Validated<VulkanError>),
    Validation(Box<ValidationError>),
    DescriptorSetCreation(Validated<VulkanError>),
}

impl std::fmt::Display for RenderPassError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vulkan(e) => write!(f, "Vulkan error: {}", e),
            Self::Validation(e) => write!(f, "Validation error: {}", e),
            Self::DescriptorSetCreation(e) => write!(f, "Descriptor set creation error: {}", e),
        }
    }
}

impl std::error::Error for RenderPassError {}

impl From<Validated<VulkanError>> for RenderPassError {
    fn from(err: Validated<VulkanError>) -> Self {
        Self::Vulkan(err)
    }
}

impl From<Box<ValidationError>> for RenderPassError {
    fn from(err: Box<ValidationError>) -> Self {
        Self::Validation(err)
    }
}

pub type RenderPassResult<T> = Result<T, RenderPassError>;

const MRT_FRAMEBUFFER_COUNT: usize = 3;
pub struct MRT {
    predepth_pipeline: Arc<GraphicsPipeline>,
    gbuffer_instanced_pipeline: Arc<GraphicsPipeline>,
    pub(crate) gbuffer_depth_view: Arc<ImageView>,
    pub(crate) gbuffer_image_views: [Arc<ImageView>; MRT_FRAMEBUFFER_COUNT],
}

impl MRT {
    pub fn new(
        predepth_pipeline: Arc<GraphicsPipeline>,
        gbuffer_instanced_pipeline: Arc<GraphicsPipeline>,
        gbuffer_depth_view: Arc<ImageView>,
        gbuffer_image_views: [Arc<ImageView>; MRT_FRAMEBUFFER_COUNT],
    ) -> Self {
        Self {
            predepth_pipeline,
            gbuffer_instanced_pipeline,
            gbuffer_depth_view,
            gbuffer_image_views,
        }
    }
}

pub struct SwapchainPass {
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    pub attachment_image_views: Vec<Arc<ImageView>>,
}

impl SwapchainPass {
    pub fn new(
        pipeline: Arc<GraphicsPipeline>,
        set: Arc<DescriptorSet>,
        attachment_image_views: Vec<Arc<ImageView>>,
    ) -> Self {
        Self {
            pipeline,
            set,
            attachment_image_views,
        }
    }
}

pub struct MRTLighting {
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    pub image_view: Arc<ImageView>,
}

impl MRTLighting {
    pub fn new(
        pipeline: Arc<GraphicsPipeline>,
        set: Arc<DescriptorSet>,
        image_view: Arc<ImageView>,
    ) -> Self {
        Self {
            pipeline,
            set,
            image_view,
        }
    }
}

pub struct Composite {
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    pub image_view: Arc<ImageView>,
}

impl Composite {
    pub fn new(
        pipeline: Arc<GraphicsPipeline>,
        set: Arc<DescriptorSet>,
        image_view: Arc<ImageView>,
    ) -> Self {
        Self {
            pipeline,
            set,
            image_view,
        }
    }
}

impl MRT {
    pub fn record_predepth_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        rcx: &RenderContext,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        mesh_streams: &HashMap<MeshHandle, MeshDrawStream>,
        cull_backfaces: Culling,
        clockwise_front_face: Winding,
    ) -> RenderPassResult<()> {
        builder
            .begin_debug_utils_label(DebugUtilsLabel {
                label_name: "Predepth Z".to_string(),
                color: [0.1, 0.1, 0.9, 1.0],
                ..Default::default()
            })?
            .begin_rendering(RenderingInfo {
                render_area_extent: rcx.scissor.extent,
                render_area_offset: rcx.scissor.offset,
                color_attachments: vec![],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(ClearValue::Depth(0.0)),
                    ..RenderingAttachmentInfo::image_view(self.gbuffer_depth_view.clone())
                }),
                ..Default::default()
            })?
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())?
            .set_scissor(0, [rcx.scissor].into_iter().collect())?
            .set_depth_compare_op(CompareOp::GreaterOrEqual)?
            .set_depth_write_enable(true)?
            .set_depth_test_enable(true)?
            .set_cull_mode(cull_backfaces.into())?
            .set_front_face(clockwise_front_face.into())?
            .bind_pipeline_graphics(self.predepth_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.predepth_pipeline.layout().clone(),
                0,
                [rcx.context_descriptor_set
                    .for_frame(rcx.current_frame)
                    .clone()]
                .to_vec(),
            )?;

        let layout = self.predepth_pipeline.layout().set_layouts()[1].clone();

        for stream in mesh_streams.values() {
            let set_1 = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(
                    1,
                    stream.transforms[rcx.frame_index()].clone(),
                )],
                [],
            )
            .map_err(RenderPassError::DescriptorSetCreation)?;

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.predepth_pipeline.layout().clone(),
                    1,
                    [set_1].to_vec(),
                )?
                .bind_vertex_buffers(0, stream.mesh.position_vertex_buffer.clone())?
                .bind_index_buffer(stream.mesh.index_buffer.clone())?;

            unsafe {
                builder.draw_indexed_indirect(stream.indirect[rcx.frame_index()].clone())?;
            }
        }

        unsafe {
            builder.end_rendering()?.end_debug_utils_label()?;
        }

        Ok(())
    }

    pub fn record_gbuffer_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        rcx: &RenderContext,
        descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
        mesh_streams: &HashMap<MeshHandle, MeshDrawStream>,
        white_image_sampler: &ImageViewSampler,
        cull_backfaces: Culling,
        clockwise_front_face: Winding,
    ) -> RenderPassResult<()> {
        builder
            .begin_debug_utils_label(DebugUtilsLabel {
                label_name: "Instanced MRT Geometry".to_string(),
                color: [0.99, 0.1, 0.1, 1.0],
                ..Default::default()
            })?
            .begin_rendering(RenderingInfo {
                render_area_extent: rcx.scissor.extent,
                render_area_offset: rcx.scissor.offset,
                color_attachments: vec![
                    Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(self.gbuffer_image_views[0].clone())
                    }),
                    Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(self.gbuffer_image_views[1].clone())
                    }),
                    Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(self.gbuffer_image_views[2].clone())
                    }),
                ],
                depth_attachment: Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Load,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: None,
                    ..RenderingAttachmentInfo::image_view(self.gbuffer_depth_view.clone())
                }),
                ..Default::default()
            })?
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())?
            .set_scissor(0, [rcx.scissor].into_iter().collect())?
            .set_depth_compare_op(CompareOp::Equal)?
            .set_depth_write_enable(true)?
            .set_depth_test_enable(true)?
            .set_cull_mode(cull_backfaces.into())?
            .set_front_face(clockwise_front_face.into())?
            .bind_pipeline_graphics(self.gbuffer_instanced_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.gbuffer_instanced_pipeline.layout().clone(),
                0,
                [rcx.context_descriptor_set
                    .for_frame(rcx.current_frame)
                    .clone()]
                .to_vec(),
            )?;

        let layout = self.gbuffer_instanced_pipeline.layout().set_layouts()[1].clone();

        for stream in mesh_streams.values() {
            let set_1 = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler_array(
                        0,
                        0,
                        stream
                            .mesh
                            .texture_array
                            .iter()
                            .cloned()
                            .map(|v| (v.view.clone(), white_image_sampler.sampler.clone()))
                            .chain(std::iter::repeat((
                                white_image_sampler.view.clone(),
                                white_image_sampler.sampler.clone(),
                            )))
                            .take(256),
                    ),
                    WriteDescriptorSet::buffer(1, stream.transforms[rcx.frame_index()].clone()),
                    WriteDescriptorSet::buffer(2, stream.material_ids.clone()),
                    WriteDescriptorSet::buffer(3, stream.mesh.materials_buffer.clone()),
                ],
                [],
            )
            .map_err(RenderPassError::DescriptorSetCreation)?;

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.gbuffer_instanced_pipeline.layout().clone(),
                    1,
                    [set_1].to_vec(),
                )?
                .bind_vertex_buffers(0, stream.mesh.vertex_buffer.clone())?
                .bind_index_buffer(stream.mesh.index_buffer.clone())?;

            unsafe {
                builder.draw_indexed_indirect(stream.indirect[rcx.frame_index()].clone())?;
            }
        }

        unsafe {
            builder.end_rendering()?.end_debug_utils_label()?;
        }

        Ok(())
    }
}

impl MRTLighting {
    pub fn record_lighting_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        rcx: &RenderContext,
    ) -> RenderPassResult<()> {
        let descriptor_sets = vec![
            rcx.context_descriptor_set
                .for_frame(rcx.current_frame)
                .clone(),
            self.set.clone(),
        ];

        builder
            .begin_debug_utils_label(DebugUtilsLabel {
                label_name: "MRT Lighting".to_string(),
                color: [0.1, 0.99, 0.9, 1.0],
                ..Default::default()
            })?
            .begin_rendering(RenderingInfo {
                render_area_extent: rcx.scissor.extent,
                render_area_offset: rcx.scissor.offset,
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(self.image_view.clone())
                })],
                ..Default::default()
            })?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_sets,
            )?;

        unsafe {
            builder.draw(3, 1, 0, 0)?;
            builder.end_rendering()?.end_debug_utils_label()?;
        }

        Ok(())
    }
}

impl Composite {
    pub fn record_composite_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        scissor: &Scissor,
        exposure: f32,
    ) -> RenderPassResult<()> {
        let descriptor_sets = vec![self.set.clone()];
        let push_constants = [exposure];

        builder
            .begin_debug_utils_label(DebugUtilsLabel {
                label_name: "Compositing".to_string(),
                color: [0.99, 0.99, 0.0, 1.0],
                ..Default::default()
            })?
            .begin_rendering(RenderingInfo {
                render_area_extent: scissor.extent,
                render_area_offset: scissor.offset,
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(self.image_view.clone())
                })],
                ..Default::default()
            })?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_sets,
            )?;

        unsafe {
            builder.draw(3, 1, 0, 0)?;
            builder.end_rendering()?.end_debug_utils_label()?;
        }

        Ok(())
    }
}

impl SwapchainPass {
    pub fn record_present_pass(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        scissor: &Scissor,
        image_index: usize,
    ) -> RenderPassResult<()> {
        let descriptor_sets = vec![self.set.clone()];

        builder
            .begin_debug_utils_label(DebugUtilsLabel {
                label_name: "Presentation".to_string(),
                color: [0.99, 0.0, 0.75, 1.0],
                ..Default::default()
            })?
            .begin_rendering(RenderingInfo {
                render_area_extent: scissor.extent,
                render_area_offset: scissor.offset,
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                    ..RenderingAttachmentInfo::image_view(
                        self.attachment_image_views[image_index].clone(),
                    )
                })],
                ..Default::default()
            })?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                descriptor_sets,
            )?;

        unsafe {
            builder.draw(3, 1, 0, 0)?;
            builder.end_rendering()?.end_debug_utils_label()?;
        }

        Ok(())
    }
}
