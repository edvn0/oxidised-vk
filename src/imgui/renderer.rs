use std::{collections::BTreeMap, sync::Arc};

use imgui::{DrawCmd, DrawData};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding,
            DescriptorSetLayoutCreateInfo, DescriptorType,
        },
    },
    device::{Device, Queue},
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo, graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::ViewportState,
        }, layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange}
    },
    shader::ShaderStages, sync::GpuFuture,
};

use crate::{imgui::shaders, mesh::ImageViewSampler};

#[repr(C)]
#[derive(BufferContents, Copy, Clone)]
pub struct OwnedDrawVert {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub col: [u8; 4],
}

struct FrameBuffers {
    vb: Subbuffer<[OwnedDrawVert]>,
    ib: Subbuffer<[u16]>,
    vtx_capacity: usize,
    idx_capacity: usize,
}

pub struct ImGuiRenderer {
    device: Arc<Device>,
    allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pipeline: Arc<GraphicsPipeline>,

    frames: Vec<FrameBuffers>,
    textures: Vec<Option<Arc<DescriptorSet>>>,

    sampler_clamp: Arc<Sampler>,
}

impl ImGuiRenderer {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        image_format: Format,
        frame_count: usize,
    ) -> Self {
        let sampler_clamp = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [
                    SamplerAddressMode::ClampToEdge,
                    SamplerAddressMode::ClampToEdge,
                    SamplerAddressMode::ClampToEdge,
                ],
                ..Default::default()
            },
        )
        .unwrap();

        let pipeline = Self::create_pipeline(device.clone(), image_format);

        let frames = (0..frame_count)
            .map(|_| Self::empty_frame(allocator.clone()))
            .collect();

        Self {
            device,
            allocator,
            descriptor_set_allocator,
            pipeline,
            frames,
            sampler_clamp,
            textures: Vec::new(),
        }
    }

    pub fn upload_font_atlas(
        &mut self,
        imgui: &mut imgui::Context,
        queue: Arc<Queue>,
        cb_allocator: Arc<StandardCommandBufferAllocator>,
    ) {
        let fonts = imgui.fonts();
        let atlas = fonts.build_rgba32_texture();

        let tex = create_image_from_rgba_bytes(
            queue,
            cb_allocator,
            self.allocator.clone(),
            atlas.width as u32,
            atlas.height as u32,
            atlas.data,
            self.sampler_clamp.clone(),
        );

        let tex_id = self.register_texture(tex);
        fonts.tex_id = imgui::TextureId::from(tex_id);
    }

    pub fn draw(
        &mut self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_data: &DrawData,
        frame_index: usize,
    ) {
        let (vb_addr, ib) = {
            let fb = &mut self.frames[frame_index];
            ensure_capacity(&self.allocator, fb, draw_data);

            {
                let mut vtx_dst = fb.vb.write().unwrap();
                let mut idx_dst = fb.ib.write().unwrap();

                let mut vtx_offset = 0;
                let mut idx_offset = 0;

                for list in draw_data.draw_lists() {
                    for (d, s) in vtx_dst[vtx_offset..vtx_offset + list.vtx_buffer().len()]
                        .iter_mut()
                        .zip(list.vtx_buffer())
                    {
                        d.pos = s.pos;
                        d.uv = s.uv;
                        d.col = s.col;
                    }

                    idx_dst[idx_offset..idx_offset + list.idx_buffer().len()]
                        .copy_from_slice(list.idx_buffer());

                    vtx_offset += list.vtx_buffer().len();
                    idx_offset += list.idx_buffer().len();
                }
            }

            (
                fb.vb.buffer().device_address().unwrap().get(),
                fb.ib.clone(),
            )
        };

        cmd.bind_pipeline_graphics(self.pipeline.clone()).unwrap();
        cmd.bind_index_buffer(ib).unwrap();

        let l = draw_data.display_pos[0];
        let r = l + draw_data.display_size[0];
        let t = draw_data.display_pos[1];
        let b = t + draw_data.display_size[1];

        let mut idx_offset = 0;
        let mut vtx_offset = 0;

        for list in draw_data.draw_lists() {
            for cmd_i in list.commands() {
                let DrawCmd::Elements { count, cmd_params } = cmd_i else { continue };

                let set = self.texture_set(cmd_params.texture_id.id() as usize);

                cmd.bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    vec![set.clone()],
                )
                .unwrap();

                let pc = shaders::vs::PC {
                    lrtb: [l, r, t, b],
                    vb: vb_addr,
                };

                cmd.push_constants(self.pipeline.layout().clone(), 0, pc)
                    .unwrap();

                unsafe {
                    cmd.draw_indexed(
                        count as u32,
                        1,
                        idx_offset + cmd_params.idx_offset as u32,
                        vtx_offset as i32 + (cmd_params.vtx_offset as i32),
                        0,
                    )
                    .unwrap();
                }
            }

            idx_offset += list.idx_buffer().len() as u32;
            vtx_offset += list.vtx_buffer().len();
        }
    }

    fn register_texture(&mut self, tex: Arc<ImageViewSampler>) -> usize {
        let layout = self.pipeline.layout().set_layouts()[0].clone();

        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout,
            [WriteDescriptorSet::image_view_sampler(
                0,
                tex.view.clone(),
                tex.sampler.clone(),
            )],
            [],
        )
        .unwrap();

        for (i, slot) in self.textures.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(set);
                return i;
            }
        }

        let id = self.textures.len();
        self.textures.push(Some(set));
        id
    }

    fn texture_set(&self, id: usize) -> &Arc<DescriptorSet> {
        self.textures[id].as_ref().unwrap()
    }

    fn empty_frame(allocator: Arc<StandardMemoryAllocator>) -> FrameBuffers {
        let vb = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            1,
        )
        .unwrap();

        let ib = Buffer::new_slice(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            1,
        )
        .unwrap();

        FrameBuffers {
            vb,
            ib,
            vtx_capacity: 1,
            idx_capacity: 1,
        }
    }

    fn create_pipeline(device: Arc<Device>, format: Format) -> Arc<GraphicsPipeline> {
        let vs = shaders::vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = shaders::fs::load(device.clone()).unwrap().entry_point("main").unwrap();

        let set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: BTreeMap::from([(
                    0,
                    {
                        let mut b = DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::CombinedImageSampler,
                        );
                        b.stages = ShaderStages::FRAGMENT;
                        b
                    },
                )]),
                ..Default::default()
            },
        )
        .unwrap();

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![set_layout],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::VERTEX,
                    offset: 0,
                    size: std::mem::size_of::<shaders::vs::PC>() as u32,
                }],
                ..Default::default()
            },
        )
        .unwrap();

        let mut ci = GraphicsPipelineCreateInfo::layout(layout);
        ci.stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ]
        .into_iter()
        .collect();

        ci.vertex_input_state = Some(VertexInputState::new());
        ci.input_assembly_state = Some(InputAssemblyState::default());
        ci.viewport_state = Some(ViewportState::default());
        ci.rasterization_state = Some(RasterizationState::default());
        ci.multisample_state = Some(MultisampleState::default());
        ci.color_blend_state = Some(ColorBlendState::with_attachment_states(
            1,
            ColorBlendAttachmentState {
                blend: Some(AttachmentBlend::alpha()),
                ..ColorBlendAttachmentState::default()
            },
        ));

        ci.subpass = Some(
            vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(format)],
                ..Default::default()
            }
            .into(),
        );

        GraphicsPipeline::new(device, None, ci).unwrap()
    }
}

fn ensure_capacity(
    allocator: &Arc<StandardMemoryAllocator>,
    fb: &mut FrameBuffers,
    draw_data: &DrawData,
) {
    let vtx = draw_data.total_vtx_count as usize;
    let idx = draw_data.total_idx_count as usize;

    if vtx > fb.vtx_capacity {
        fb.vb = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vtx as u64,
        )
        .unwrap();
        fb.vtx_capacity = vtx;
    }

    if idx > fb.idx_capacity {
        fb.ib = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            idx as u64,
        )
        .unwrap();
        fb.idx_capacity = idx;
    }
}

fn create_image_from_rgba_bytes(
    queue: Arc<Queue>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    allocator: Arc<StandardMemoryAllocator>,
    width: u32,
    height: u32,
    rgba: &[u8],
    sampler: Arc<Sampler>,
) -> Arc<ImageViewSampler> {
    let image = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [width, height, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();

    let staging = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        rgba.iter().copied(),
    )
    .unwrap();

    {
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator,
            queue.queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging,
                image.clone(),
            ))
            .unwrap();

        builder
            .build()
            .unwrap()
            .execute(queue)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    Arc::new(ImageViewSampler::new(
        ImageView::new_default(image).unwrap(),
        sampler,
    ))
}
