use crate::{
    image::{ImageDimensions, ImageInfo, create_image},
    imgui::shaders,
    mesh::ImageViewSampler,
};

use imgui::{DrawCmd, DrawData, FontConfig, FontSource};
use std::{collections::BTreeMap, sync::Arc};

use vulkano::{
    buffer::{
        BufferContents, BufferUsage, Subbuffer,
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
    },
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
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
    image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::VertexInputState,
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange},
    },
    shader::ShaderStages,
};

use vulkano::pipeline::DynamicState;
use vulkano::pipeline::DynamicState::{DepthTestEnable, DepthWriteEnable};

#[repr(C)]
#[derive(BufferContents, Copy, Clone)]
pub struct OwnedDrawVert {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub col: [u8; 4],
}

pub struct ImGuiRenderer {
    allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    pipeline: Arc<GraphicsPipeline>,
    texture_set: Arc<DescriptorSet>,

    textures: Vec<Option<Arc<ImageViewSampler>>>,
    free_texture_ids: Vec<usize>,

    sampler_clamp: Arc<Sampler>,

    vtx_alloc: SubbufferAllocator,
    idx_alloc: SubbufferAllocator,
}

impl ImGuiRenderer {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        image_format: Format,
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

        let (pipeline, texture_set) = Self::create_pipeline(
            descriptor_set_allocator.clone(),
            device.clone(),
            image_format,
            sampler_clamp.clone(),
        );

        let vtx_alloc = SubbufferAllocator::new(
            allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER
                    | BufferUsage::STORAGE_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::PREFER_DEVICE,
                arena_size: 1024 * 1024,
                ..Default::default()
            },
        );

        let idx_alloc = SubbufferAllocator::new(
            allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDEX_BUFFER,
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS
                    | MemoryTypeFilter::PREFER_DEVICE,
                arena_size: 512 * 1024,
                ..Default::default()
            },
        );

        Self {
            allocator,
            descriptor_set_allocator,
            pipeline,
            texture_set,
            textures: Vec::new(),
            free_texture_ids: Vec::new(),
            sampler_clamp,
            vtx_alloc,
            idx_alloc,
        }
    }

    pub fn upload_font_atlas(
        &mut self,
        imgui: &mut imgui::Context,
        queue: Arc<Queue>,
        cb_allocator: Arc<vulkano::command_buffer::allocator::StandardCommandBufferAllocator>,
    ) {
        const ROBOTO_REGULAR: &[u8] = include_bytes!("../../assets/fonts/Roboto-Regular.ttf");
        const ROBOTO_BLACK: &[u8] = include_bytes!("../../assets/fonts/Roboto-Black.ttf");

        let fonts = [
            ("roboto_regular", ROBOTO_REGULAR),
            ("roboto_black", ROBOTO_BLACK),
        ];

        let base_config = FontConfig {
            oversample_h: 2,
            oversample_v: 2,
            pixel_snap_h: true,
            rasterizer_multiply: 1.1,
            glyph_offset: [0.0, 0.0],
            ..Default::default()
        };

        let size_range = 8..=15;

        let atlas = imgui.fonts();

        for &(name, font_data) in &fonts {
            for size in size_range.clone() {
                atlas.add_font(&[FontSource::TtfData {
                    data: font_data,
                    size_pixels: size as f32,
                    config: Some(FontConfig {
                        name: Some(name.to_string()),
                        ..base_config.clone()
                    }),
                }]);
            }
        }

        let fonts = imgui.fonts();
        let atlas = fonts.build_rgba32_texture();

        let image_info = ImageInfo::new(
            ImageDimensions::Dim2d([atlas.width as u32, atlas.height as u32]),
            Format::R8G8B8A8_UNORM,
            None,
            "ImGui Font Atlas".into(),
            self.sampler_clamp.clone(),
        );

        let image = create_image(
            queue,
            cb_allocator,
            self.allocator.clone(),
            &atlas.data,
            image_info,
        );

        let id = self.register_texture(image);
        fonts.tex_id = imgui::TextureId::from(id);
    }

    pub fn draw(
        &self,
        cmd: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        draw_data: &DrawData,
        vp: (&Viewport, &Scissor),
    ) {
        if draw_data.total_vtx_count == 0 || draw_data.total_idx_count == 0 {
            return;
        }

        let vtx_buf: Subbuffer<[OwnedDrawVert]> = self
            .vtx_alloc
            .allocate_slice(draw_data.total_vtx_count as u64)
            .unwrap();
        let idx_buf: Subbuffer<[u16]> = self
            .idx_alloc
            .allocate_slice(draw_data.total_idx_count as u64)
            .unwrap();

        {
            let mut vtx = vtx_buf.write().unwrap();
            let mut idx = idx_buf.write().unwrap();

            let mut vo = 0;
            let mut io = 0;

            for list in draw_data.draw_lists() {
                for (dst, src) in vtx[vo..vo + list.vtx_buffer().len()]
                    .iter_mut()
                    .zip(list.vtx_buffer())
                {
                    dst.pos = src.pos;
                    dst.uv = src.uv;
                    dst.col = src.col;
                }

                idx[io..io + list.idx_buffer().len()].copy_from_slice(list.idx_buffer());

                vo += list.vtx_buffer().len();
                io += list.idx_buffer().len();
            }
        }

        let vb_addr = vtx_buf.device_address().unwrap();

        cmd.bind_pipeline_graphics(self.pipeline.clone()).unwrap();
        cmd.bind_index_buffer(idx_buf).unwrap();
        cmd.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipeline.layout().clone(),
            0,
            [self.texture_set.clone()].to_vec(),
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
                    _pad: 0,
                };

                let (viewport, scissor) = vp;

                cmd.push_constants(self.pipeline.layout().clone(), 0, pc)
                    .unwrap()
                    .set_depth_test_enable(false)
                    .unwrap()
                    .set_depth_write_enable(false)
                    .unwrap()
                    .set_scissor(0, [scissor.clone()].into_iter().collect())
                    .unwrap()
                    .set_viewport(0, [viewport.clone()].into_iter().collect())
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
                        b.descriptor_count = 1024;
                        b.stages = ShaderStages::FRAGMENT;
                        b.binding_flags = DescriptorBindingFlags::PARTIALLY_BOUND
                            | DescriptorBindingFlags::UPDATE_AFTER_BIND;
                        b
                    }),
                    (1, {
                        let mut b =
                            DescriptorSetLayoutBinding::descriptor_type(DescriptorType::Sampler);
                        b.stages = ShaderStages::FRAGMENT;
                        b.immutable_samplers = vec![sampler_clamp];
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

        ci.dynamic_state.insert(DepthTestEnable);
        ci.dynamic_state.insert(DynamicState::Viewport);
        ci.dynamic_state.insert(DynamicState::Scissor);
        ci.dynamic_state.insert(DepthWriteEnable);

        ci.subpass = Some(
            PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(format)],
                ..Default::default()
            }
            .into(),
        );

        let pipeline = GraphicsPipeline::new(device.clone(), None, ci).unwrap();
        let texture_set = DescriptorSet::new(descriptor_set_allocator, set_layout, [], []).unwrap();

        (pipeline, texture_set)
    }
}
