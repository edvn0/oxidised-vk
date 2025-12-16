use std::{collections::BTreeMap, sync::Arc};

use imgui::{DrawCmd, DrawData};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CopyBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
            DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType,
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
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::ViewportState,
        },
        layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
    },
    shader::ShaderStages,
    sync::GpuFuture,
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
    vb_gpu: Subbuffer<[OwnedDrawVert]>,
    ib_gpu: Subbuffer<[u16]>,

    vb_staging: Subbuffer<[OwnedDrawVert]>,
    ib_staging: Subbuffer<[u16]>,

    vtx_capacity: usize,
    idx_capacity: usize,
}

pub struct ImGuiRenderer {
    allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    pipeline: Arc<GraphicsPipeline>,
    texture_set: Arc<DescriptorSet>,

    textures: Vec<Option<Arc<ImageViewSampler>>>,
    free_texture_ids: Vec<usize>,

    sampler_clamp: Arc<Sampler>,

    frames: Vec<FrameBuffers>,
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

        let (pipeline, set) = Self::create_pipeline(
            descriptor_set_allocator.clone(),
            device.clone(),
            image_format,
            sampler_clamp.clone(),
        );

        let frames = (0..frame_count)
            .map(|_| Self::empty_frame(allocator.clone()))
            .collect();

        Self {
            allocator,
            descriptor_set_allocator,
            pipeline,
            texture_set: set,
            free_texture_ids: Vec::new(),
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

    pub fn upload(
        &mut self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_data: &DrawData,
        frame_index: usize,
    ) {
        let fb = &mut self.frames[frame_index];

        ensure_capacity(&self.allocator, fb, draw_data);
        upload_buffers(cmd, fb, draw_data);
    }

    pub fn draw(
        &self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_data: &DrawData,
        frame_index: usize,
    ) {
        let fb = &self.frames[frame_index];

        let vb_addr = fb.vb_gpu.device_address().unwrap();
        let ib = fb.ib_gpu.clone();

        cmd.bind_pipeline_graphics(self.pipeline.clone()).unwrap();
        cmd.bind_index_buffer(ib).unwrap();
        cmd.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipeline.layout().clone(),
            0,
            vec![self.texture_set.clone()],
        )
        .unwrap();

        let l = draw_data.display_pos[0];
        let r = l + draw_data.display_size[0];
        let t = draw_data.display_pos[1];
        let b = t + draw_data.display_size[1];

        let mut idx_offset = 0;
        let mut vtx_offset = 0;

        for list in draw_data.draw_lists() {
            for cmd_i in list.commands() {
                let DrawCmd::Elements { count, cmd_params } = cmd_i else {
                    continue;
                };

                let pc = shaders::vs::PC {
                    lrtb: [l, r, t, b],
                    vb: vb_addr.get(),
                    texture_index: cmd_params.texture_id.id() as u32,
                };

                cmd.push_constants(self.pipeline.layout().clone(), 0, pc)
                    .unwrap()
                    .set_depth_test_enable(false)
                    .unwrap()
                    .set_depth_write_enable(false)
                    .unwrap();
                unsafe {
                    cmd.draw_indexed(
                        count as u32,
                        1,
                        idx_offset + cmd_params.idx_offset as u32,
                        vtx_offset as i32 + cmd_params.vtx_offset as i32,
                        0,
                    )
                    .unwrap();
                }
            }

            idx_offset += list.idx_buffer().len() as u32;
            vtx_offset += list.vtx_buffer().len();
        }
    }

    fn rebuild_texture_set(&mut self) {
        let writes = self
            .textures
            .iter()
            .enumerate()
            .filter_map(|(i, tex)| {
                tex.as_ref()
                    .map(|t| WriteDescriptorSet::image_view_array(0, i as u32, [t.view.clone()]))
            })
            .collect::<Vec<_>>();

        self.texture_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            self.texture_set.layout().clone(),
            writes,
            [],
        )
        .unwrap();
    }

    fn register_texture(&mut self, tex: Arc<ImageViewSampler>) -> usize {
        let id = self.free_texture_ids.pop().unwrap_or_else(|| {
            let id = self.textures.len();
            self.textures.push(None);
            id
        });

        self.textures[id] = Some(tex);
        self.rebuild_texture_set();
        id
    }


    fn empty_frame(allocator: Arc<StandardMemoryAllocator>) -> FrameBuffers {
        let ib = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            1,
        )
        .unwrap();

        let vb_staging = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            1,
        )
        .unwrap();

        let vb_gpu = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            1,
        )
        .unwrap();

        let ib_staging = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
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
            vb_gpu,
            ib_gpu: ib,
            vb_staging,
            ib_staging,
            vtx_capacity: 1,
            idx_capacity: 1,
        }
    }

    fn create_pipeline(
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        device: Arc<Device>,
        format: Format,
        sampler_clamp: Arc<Sampler>,
    ) -> (Arc<GraphicsPipeline>, Arc<DescriptorSet>) {
        let vs = shaders::vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = shaders::fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                flags: DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                bindings: BTreeMap::from([
                    (0, {
                        let mut b = DescriptorSetLayoutBinding::descriptor_type(
                            DescriptorType::SampledImage,
                        );
                        const MAX_IMGUI_TEXTURES: u32 = 1024;
                        b.descriptor_count = MAX_IMGUI_TEXTURES;
                        b.stages = ShaderStages::FRAGMENT;
                        b.binding_flags = DescriptorBindingFlags::PARTIALLY_BOUND
                            | DescriptorBindingFlags::UPDATE_AFTER_BIND;
                        b
                    }),
                    (1, {
                        let mut b =
                            DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler);
                        b.descriptor_count = 1;
                        b.stages = ShaderStages::FRAGMENT;
                        b.immutable_samplers = vec![sampler_clamp.clone()];
                        b
                    }),
                ]),
                ..Default::default()
            },
        )
        .unwrap();

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![set_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
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

        let pipeline = GraphicsPipeline::new(device.clone(), None, ci).unwrap();

        let texture_set =
            DescriptorSet::new(descriptor_set_allocator.clone(), set_layout, [], []).unwrap();

        (pipeline, texture_set)
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
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
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

fn upload_buffers(
    cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    fb: &FrameBuffers,
    draw_data: &DrawData,
) {
    let mut vtx = fb.vb_staging.write().unwrap();
    let mut idx = fb.ib_staging.write().unwrap();

    let mut vo = 0;
    let mut io = 0;

    for list in draw_data.draw_lists() {
        for (d, s) in vtx[vo..vo + list.vtx_buffer().len()]
            .iter_mut()
            .zip(list.vtx_buffer())
        {
            d.pos = s.pos;
            d.uv = s.uv;
            d.col = s.col;
        }

        idx[io..io + list.idx_buffer().len()].copy_from_slice(list.idx_buffer());

        vo += list.vtx_buffer().len();
        io += list.idx_buffer().len();
    }

    cmd.copy_buffer(CopyBufferInfo::buffers(
        fb.vb_staging.clone(),
        fb.vb_gpu.clone(),
    ))
    .unwrap();
    cmd.copy_buffer(CopyBufferInfo::buffers(
        fb.ib_staging.clone(),
        fb.ib_gpu.clone(),
    ))
    .unwrap();
}

fn ensure_capacity(
    allocator: &Arc<StandardMemoryAllocator>,
    fb: &mut FrameBuffers,
    draw_data: &DrawData,
) {
    let total_vtx = draw_data.total_vtx_count as usize;
    let total_idx = draw_data.total_idx_count as usize;

    if total_vtx > fb.vtx_capacity {
        fb.vb_staging = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            total_vtx as u64,
        )
        .unwrap();

        fb.vb_gpu = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            total_vtx as u64,
        )
        .unwrap();

        fb.vtx_capacity = total_vtx;
    }

    if total_idx > fb.idx_capacity {
        fb.ib_staging = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            total_idx as u64,
        )
        .unwrap();

        fb.ib_gpu = Buffer::new_slice(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            total_idx as u64,
        )
        .unwrap();

        fb.idx_capacity = total_idx;
    }
}
