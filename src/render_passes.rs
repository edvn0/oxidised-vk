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
        graphics::{
            depth_stencil::CompareOp,
            viewport::{Scissor, Viewport},
        },
        *,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
};

use crate::{
    imgui::renderer::ImGuiRenderer,
    mesh::ImageViewSampler,
    mesh_registry::MeshHandle,
    render_context::{Culling, MeshDrawStream, Winding},
};

pub mod data {
    use super::*;

    pub struct FrameContext<'a> {
        pub frame_index: usize,
        pub viewport: &'a Viewport,
        pub scissor: &'a Scissor,
        pub culling: Culling,
        pub winding: Winding,
    }

    pub struct RenderResources<'a> {
        pub mesh_streams: &'a HashMap<MeshHandle, MeshDrawStream>,
        pub descriptor_sets: &'a Arc<StandardDescriptorSetAllocator>,
        pub white_sampler: &'a ImageViewSampler,
        pub swapchain_views: &'a [Arc<ImageView>],
        pub viewport: &'a Viewport,
        pub scissor: &'a Scissor,
        pub current_frame: usize,
        pub frame_descriptor_set: Arc<DescriptorSet>,
    }

    #[derive(Debug)]
    pub enum RenderPassError {
        Vulkan(Validated<VulkanError>),
        Validation(Box<ValidationError>),
        DescriptorSetCreation(Validated<VulkanError>),
    }

    impl std::fmt::Display for RenderPassError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Vulkan(e) => write!(f, "Vulkan error: {e}"),
                Self::Validation(e) => write!(f, "Validation error: {e}"),
                Self::DescriptorSetCreation(e) => {
                    write!(f, "Descriptor set creation error: {e}")
                }
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
}

pub mod recorder {
    use super::*;

    pub struct RenderRecorder<'a> {
        pub cmd: &'a mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        pub imgui_context: &'a mut imgui::Context,
        pub imgui_renderer: &'a mut ImGuiRenderer,
    }
}

pub mod passes {
    use crate::{
        bloom_pass::{BloomPass, BloomSettings},
        render_passes::{
            data::{FrameContext, RenderPassResult, RenderResources},
            recorder::*,
            recordings::{Composite, CompositeSettings, MRT, MRTLighting, SwapchainPass},
        },
    };

    use super::*;

    pub trait RenderPass {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            frame: &FrameContext,
            resources: &RenderResources,
        ) -> RenderPassResult<()>;
    }

    pub struct PreDepthPass<'a> {
        pub mrt: &'a MRT,
    }

    impl<'a> RenderPass for PreDepthPass<'a> {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            frame: &FrameContext,
            res: &RenderResources,
        ) -> RenderPassResult<()> {
            self.mrt.record_predepth_pass(
                recorder.cmd,
                res.viewport,
                res.scissor,
                res.current_frame,
                res.frame_descriptor_set.clone(),
                res.descriptor_sets,
                res.mesh_streams,
                frame.culling,
                frame.winding,
            )
        }
    }

    pub struct GBufferPass<'a> {
        pub mrt: &'a MRT,
    }

    impl<'a> RenderPass for GBufferPass<'a> {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            frame: &FrameContext,
            res: &RenderResources,
        ) -> RenderPassResult<()> {
            self.mrt.record_gbuffer_pass(
                recorder.cmd,
                res.scissor,
                res.viewport,
                res.descriptor_sets,
                res.frame_descriptor_set.clone(),
                res.mesh_streams,
                res.white_sampler,
                res.current_frame,
                frame.culling,
                frame.winding,
            )
        }
    }

    pub struct MRTLightingPass<'a> {
        pub lighting: &'a MRTLighting,
    }

    impl<'a> RenderPass for MRTLightingPass<'a> {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            frame: &FrameContext,
            res: &RenderResources,
        ) -> RenderPassResult<()> {
            self.lighting.record_lighting_pass(
                recorder.cmd,
                frame.scissor,
                res.frame_descriptor_set.clone(),
            )
        }
    }

    pub struct BloomEffectPass<'a> {
        pub bloom: &'a BloomPass,
        pub settings: &'a BloomSettings,
        pub input_image: &'a Arc<ImageView>,
    }

    impl<'a> RenderPass for BloomEffectPass<'a> {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            _frame: &FrameContext,
            res: &RenderResources,
        ) -> RenderPassResult<()> {
            if !self.settings.enabled {
                return Ok(());
            }

            self.bloom.run(
                recorder.cmd,
                res.descriptor_sets,
                self.input_image.clone(),
                self.settings,
            );

            Ok(())
        }
    }

    pub struct CompositePass<'a> {
        pub composite: &'a Composite,
        pub bloom_enabled: bool,
        pub settings: &'a CompositeSettings,
    }

    impl<'a> RenderPass for CompositePass<'a> {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            frame: &FrameContext,
            _res: &RenderResources,
        ) -> RenderPassResult<()> {
            self.composite.record_composite_pass(
                recorder.cmd,
                frame.scissor,
                self.bloom_enabled,
                self.settings,
            )
        }
    }

    pub struct PresentPass<'a> {
        pub swapchain: &'a SwapchainPass,
        pub image_index: usize,
    }

    impl<'a> RenderPass for PresentPass<'a> {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            frame: &FrameContext,
            _res: &RenderResources,
        ) -> RenderPassResult<()> {
            self.swapchain
                .record_present_pass(recorder.cmd, frame.scissor, self.image_index)
        }
    }

    pub struct ImGuiPass {
        pub image_index: usize,
    }

    impl RenderPass for ImGuiPass {
        fn record(
            &self,
            recorder: &mut RenderRecorder,
            frame: &FrameContext,
            res: &RenderResources,
        ) -> RenderPassResult<()> {
            let draw_data = recorder.imgui_context.render();

            recorder.cmd.begin_debug_utils_label(DebugUtilsLabel {
                label_name: "ImGui".into(),
                color: [0.5, 0.5, 0.9, 1.0],
                ..Default::default()
            })?;

            recorder.cmd.begin_rendering(RenderingInfo {
                render_area_extent: frame.scissor.extent,
                render_area_offset: frame.scissor.offset,
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Load,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: None,
                    ..RenderingAttachmentInfo::image_view(
                        res.swapchain_views[self.image_index].clone(),
                    )
                })],
                ..Default::default()
            })?;

            recorder
                .imgui_renderer
                .draw(recorder.cmd, draw_data, (frame.viewport, frame.scissor));

            unsafe {
                recorder.cmd.end_rendering()?;
                recorder.cmd.end_debug_utils_label()?;
            }

            Ok(())
        }
    }
}

pub mod recordings {
    use derive_more::Constructor;

    use crate::{
        engine_shaders,
        imgui::settings,
        render_passes::data::{RenderPassError, RenderPassResult},
    };

    use super::*;

    const MRT_FRAMEBUFFER_COUNT: usize = 3;

    #[derive(Constructor)]
    pub struct MRT {
        pub predepth_pipeline: Arc<GraphicsPipeline>,
        pub gbuffer_instanced_pipeline: Arc<GraphicsPipeline>,
        pub gbuffer_depth_view: Arc<ImageView>,
        pub gbuffer_image_views: [Arc<ImageView>; MRT_FRAMEBUFFER_COUNT],
    }

    #[derive(Constructor)]
    pub struct MRTLighting {
        pub pipeline: Arc<GraphicsPipeline>,
        pub set: Arc<DescriptorSet>,
        pub image_view: Arc<ImageView>,
    }

    pub struct CompositeSettings {
        pub exposure: f32,
        pub bloom_strength: f32,
    }

    impl Default for CompositeSettings {
        fn default() -> Self {
            Self {
                exposure: 1.0,
                bloom_strength: 1.0,
            }
        }
    }

    #[derive(Constructor)]
    pub struct Composite {
        pub pipeline: Arc<GraphicsPipeline>,
        pub set: Arc<DescriptorSet>,
        pub disabled_set: Arc<DescriptorSet>,
        pub image_view: Arc<ImageView>,
    }

    #[derive(Constructor)]
    pub struct SwapchainPass {
        pub pipeline: Arc<GraphicsPipeline>,
        pub set: Arc<DescriptorSet>,
        pub attachment_image_views: Vec<Arc<ImageView>>,
    }

    impl MRT {
        pub fn record_predepth_pass(
            &self,
            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
            viewport: &Viewport,
            scissor: &Scissor,
            current_frame: usize,
            frame_descriptor_set: Arc<DescriptorSet>,
            descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
            mesh_streams: &HashMap<MeshHandle, MeshDrawStream>,
            culling: Culling,
            winding: Winding,
        ) -> RenderPassResult<()> {
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Predepth".into(),
                    color: [0.1, 0.1, 0.9, 1.0],
                    ..Default::default()
                })?
                .begin_rendering(RenderingInfo {
                    render_area_extent: scissor.extent,
                    render_area_offset: scissor.offset,
                    color_attachments: vec![],
                    depth_attachment: Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some(ClearValue::Depth(0.0)),
                        ..RenderingAttachmentInfo::image_view(self.gbuffer_depth_view.clone())
                    }),
                    ..Default::default()
                })?
                .set_viewport(0, [viewport.clone()].into_iter().collect())?
                .set_scissor(0, [scissor.clone()].into_iter().collect())?
                .set_depth_compare_op(CompareOp::GreaterOrEqual)?
                .set_depth_write_enable(true)?
                .set_depth_test_enable(true)?
                .set_cull_mode(culling.into())?
                .set_front_face(winding.into())?
                .bind_pipeline_graphics(self.predepth_pipeline.clone())?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.predepth_pipeline.layout().clone(),
                    0,
                    [frame_descriptor_set].to_vec(),
                )?;

            let layout = self.predepth_pipeline.layout().set_layouts()[1].clone();

            for stream in mesh_streams.values() {
                let set_1 = DescriptorSet::new(
                    descriptor_set_allocator.clone(),
                    layout.clone(),
                    [WriteDescriptorSet::buffer(
                        1,
                        stream.transforms[current_frame].clone(),
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
                    builder.draw_indexed_indirect(stream.indirect[current_frame].clone())?;
                }
            }

            unsafe {
                builder.end_rendering()?;
                builder.end_debug_utils_label()?;
            }

            Ok(())
        }

        pub fn record_gbuffer_pass(
            &self,
            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
            scissor: &Scissor,
            viewport: &Viewport,
            descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
            descriptor_set: Arc<DescriptorSet>,
            mesh_streams: &HashMap<MeshHandle, MeshDrawStream>,
            white_sampler: &ImageViewSampler,
            current_frame: usize,
            culling: Culling,
            winding: Winding,
        ) -> RenderPassResult<()> {
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "GBuffer".into(),
                    color: [0.9, 0.1, 0.1, 1.0],
                    ..Default::default()
                })?
                .begin_rendering(RenderingInfo {
                    render_area_extent: scissor.extent,
                    render_area_offset: scissor.offset,
                    color_attachments: self
                        .gbuffer_image_views
                        .iter()
                        .map(|view| {
                            Some(RenderingAttachmentInfo {
                                load_op: AttachmentLoadOp::Clear,
                                store_op: AttachmentStoreOp::Store,
                                clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                                ..RenderingAttachmentInfo::image_view(view.clone())
                            })
                        })
                        .collect(),
                    depth_attachment: Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Load,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: None,
                        ..RenderingAttachmentInfo::image_view(self.gbuffer_depth_view.clone())
                    }),
                    ..Default::default()
                })?
                .set_viewport(0, [viewport.clone()].into_iter().collect())?
                .set_scissor(0, [scissor.clone()].into_iter().collect())?
                .set_depth_compare_op(CompareOp::Equal)?
                .set_depth_write_enable(true)?
                .set_depth_test_enable(true)?
                .set_cull_mode(culling.into())?
                .set_front_face(winding.into())?
                .bind_pipeline_graphics(self.gbuffer_instanced_pipeline.clone())?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.gbuffer_instanced_pipeline.layout().clone(),
                    0,
                    [descriptor_set].to_vec(),
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
                                .map(|v| (v.view.clone(), white_sampler.sampler.clone()))
                                .chain(std::iter::repeat((
                                    white_sampler.view.clone(),
                                    white_sampler.sampler.clone(),
                                )))
                                .take(256),
                        ),
                        WriteDescriptorSet::buffer(1, stream.transforms[current_frame].clone()),
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
                    builder.draw_indexed_indirect(stream.indirect[current_frame].clone())?;
                }
            }

            unsafe {
                builder.end_rendering()?;
                builder.end_debug_utils_label()?;
            }

            Ok(())
        }
    }

    impl MRTLighting {
        pub fn record_lighting_pass(
            &self,
            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
            scissor: &Scissor,
            frame_set: Arc<DescriptorSet>,
        ) -> RenderPassResult<()> {
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Lighting".into(),
                    color: [0.1, 0.9, 0.9, 1.0],
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
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    vec![frame_set, self.set.clone()],
                )?;

            unsafe {
                builder.draw(3, 1, 0, 0)?;
                builder.end_rendering()?;
                builder.end_debug_utils_label()?;
            }

            Ok(())
        }
    }

    impl Composite {
        pub fn record_composite_pass(
            &self,
            builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
            scissor: &Scissor,
            bloom_enabled: bool,
            settings: &CompositeSettings,
        ) -> RenderPassResult<()> {
            let set = if bloom_enabled {
                &self.set
            } else {
                &self.disabled_set
            };

            let pc = engine_shaders::composite::fs::PC {
                exposure: settings.exposure,
                bloom_strength: settings.bloom_strength,
            };

            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Composite".into(),
                    color: [0.9, 0.9, 0.0, 1.0],
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
                .push_constants(self.pipeline.layout().clone(), 0, pc)?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    vec![set.clone()],
                )?;

            unsafe {
                builder.draw(3, 1, 0, 0)?;
                builder.end_rendering()?;
                builder.end_debug_utils_label()?;
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
            builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Present".into(),
                    color: [0.9, 0.0, 0.7, 1.0],
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
                    vec![self.set.clone()],
                )?;

            unsafe {
                builder.draw(3, 1, 0, 0)?;
                builder.end_rendering()?;
                builder.end_debug_utils_label()?;
            }

            Ok(())
        }
    }
}
