use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
};
use vulkano::shader::ShaderStages;

struct BloomStage {
    pipeline: Arc<ComputePipeline>,
    layout: Arc<DescriptorSetLayout>,
}

pub struct BloomPass {
    extract: BloomStage,
    downsample: BloomStage,
    upsample: BloomStage,
    sampler: Arc<Sampler>,
    mip_views: Vec<Arc<ImageView>>,
    mip_sizes: Vec<(u32, u32)>,
    mip_count: usize,
}

fn compute_mip_count(width: u32, height: u32) -> usize {
    let min_dim = width.min(height) as f32;
    let log2_v = min_dim.log2();
    let mips = log2_v.floor() as usize - 2;
    mips.max(1)
}

pub mod bloom_limits {
    pub const THRESHOLD_MIN: f32 = 0.0;
    pub const THRESHOLD_MAX: f32 = 5.0;
    pub const THRESHOLD_DEFAULT: f32 = 1.0;

    pub const INTENSITY_MIN: f32 = 0.0;
    pub const INTENSITY_MAX: f32 = 3.0;
    pub const INTENSITY_DEFAULT: f32 = 1.0;

    pub const FILTER_RADIUS_MIN: f32 = 0.0005;
    pub const FILTER_RADIUS_MAX: f32 = 0.02;
    pub const FILTER_RADIUS_DEFAULT: f32 = 0.005;
}

pub struct BloomSettings {
    pub threshold: f32,
    pub intensity: f32,
    pub filter_radius: f32,
    pub enabled: bool,
}

impl Default for BloomSettings {
    fn default() -> Self {
        Self {
            threshold: bloom_limits::THRESHOLD_DEFAULT,
            intensity: bloom_limits::INTENSITY_DEFAULT,
            filter_radius: bloom_limits::FILTER_RADIUS_DEFAULT,
            enabled: true,
        }
    }
}

impl BloomPass {
    pub fn new(
        device: &Arc<Device>,
        allocator: &Arc<StandardMemoryAllocator>,
        width: u32,
        height: u32,
    ) -> Result<Self, ()> {
        if width == 0 || height == 0 {
            return Err(());
        }

        let mip_count = compute_mip_count(width, height);

        let mut mip_sizes = Vec::<(u32, u32)>::with_capacity(mip_count);
        let mut w = width;
        let mut h = height;
        for _ in 0..mip_count {
            mip_sizes.push((w, h));
            w = (w / 2).max(1);
            h = (h / 2).max(1);
        }

        let create_image = |allocator: &Arc<StandardMemoryAllocator>, w: u32, h: u32| {
            let img = Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [w, h, 1],
                    usage: ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap();
            ImageView::new_default(img).unwrap()
        };

        let mut mip_views: Vec<Arc<ImageView>> = Vec::with_capacity(mip_count);
        (0..mip_count).for_each(|i| {
            let (mw, mh) = mip_sizes[i];
            mip_views.push(create_image(allocator, mw, mh));
        });

        // Linear sampler with clamp-to-edge for bilinear filtering
        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..Default::default()
            },
        )
        .unwrap();

        // Descriptor layout for extract (2 storage images)
        let make_storage_layout = |device: &Arc<Device>, pc_size: u32| {
            let mut info = DescriptorSetLayoutCreateInfo::default();
            info.bindings.insert(0, {
                let mut b =
                    DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage);
                b.stages = ShaderStages::COMPUTE;
                b
            });
            info.bindings.insert(1, {
                let mut b =
                    DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage);
                b.stages = ShaderStages::COMPUTE;
                b
            });

            let desc_layout = DescriptorSetLayout::new(device.clone(), info).unwrap();

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![desc_layout.clone()],
                    push_constant_ranges: if pc_size > 0 {
                        vec![PushConstantRange {
                            stages: ShaderStages::COMPUTE,
                            offset: 0,
                            size: pc_size,
                        }]
                    } else {
                        vec![]
                    },
                    ..Default::default()
                },
            )
            .unwrap();

            (desc_layout, layout)
        };

        // Descriptor layout for downsample/upsample (sampler + sampled image + storage image)
        let make_sampler_layout = |device: &Arc<Device>, pc_size: u32| {
            let mut info = DescriptorSetLayoutCreateInfo::default();
            // binding 0: combined image sampler (src)
            info.bindings.insert(0, {
                let mut b = DescriptorSetLayoutBinding::descriptor_type(
                    DescriptorType::CombinedImageSampler,
                );
                b.stages = ShaderStages::COMPUTE;
                b
            });
            // binding 1: storage image (dst)
            info.bindings.insert(1, {
                let mut b =
                    DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage);
                b.stages = ShaderStages::COMPUTE;
                b
            });

            let desc_layout = DescriptorSetLayout::new(device.clone(), info).unwrap();

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![desc_layout.clone()],
                    push_constant_ranges: if pc_size > 0 {
                        vec![PushConstantRange {
                            stages: ShaderStages::COMPUTE,
                            offset: 0,
                            size: pc_size,
                        }]
                    } else {
                        vec![]
                    },
                    ..Default::default()
                },
            )
            .unwrap();

            (desc_layout, layout)
        };

        // Extract shader - threshold bright pixels
        mod cs_extract {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 450
                    layout(local_size_x = 8, local_size_y = 8) in;
                    layout(set=0, binding=0, rgba16f) readonly uniform image2D hdr_in;
                    layout(set=0, binding=1, rgba16f) writeonly uniform image2D bloom_out;

                    layout(push_constant) uniform Push {
                        float threshold;
                    } pc;

                    void main() {
                        ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
                        ivec2 size = imageSize(bloom_out);
                        if (uv.x >= size.x || uv.y >= size.y) return;

                        vec4 c = imageLoad(hdr_in, uv);
                        float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
                        vec4 result = l > pc.threshold ? c : vec4(0.0);
                        result = max(result, 0.0001); // Prevent black artifacts
                        imageStore(bloom_out, uv, result);
                    }
                "
            }
        }

        // CoD: Advanced Warfare 13-tap downsample filter
        mod cs_downsample {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 450
                    layout(local_size_x = 8, local_size_y = 8) in;
                    layout(set=0, binding=0) uniform sampler2D src;
                    layout(set=0, binding=1, rgba16f) writeonly uniform image2D dst;

                    layout(push_constant) uniform Push {
                        vec2 src_resolution;
                    } pc;

                    void main() {
                        ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
                        ivec2 size = imageSize(dst);
                        if (uv.x >= size.x || uv.y >= size.y) return;

                        vec2 texel_size = 1.0 / pc.src_resolution;
                        float x = texel_size.x;
                        float y = texel_size.y;

                        // Center of destination pixel in source texture coordinates
                        vec2 tex_coord = (vec2(uv) + 0.5) / vec2(size);

                        // 13-tap filter pattern (CoD: AW)
                        // a - b - c
                        // - j - k -
                        // d - e - f
                        // - l - m -
                        // g - h - i
                        vec3 a = texture(src, vec2(tex_coord.x - 2*x, tex_coord.y + 2*y)).rgb;
                        vec3 b = texture(src, vec2(tex_coord.x,       tex_coord.y + 2*y)).rgb;
                        vec3 c = texture(src, vec2(tex_coord.x + 2*x, tex_coord.y + 2*y)).rgb;

                        vec3 d = texture(src, vec2(tex_coord.x - 2*x, tex_coord.y)).rgb;
                        vec3 e = texture(src, vec2(tex_coord.x,       tex_coord.y)).rgb;
                        vec3 f = texture(src, vec2(tex_coord.x + 2*x, tex_coord.y)).rgb;

                        vec3 g = texture(src, vec2(tex_coord.x - 2*x, tex_coord.y - 2*y)).rgb;
                        vec3 h = texture(src, vec2(tex_coord.x,       tex_coord.y - 2*y)).rgb;
                        vec3 i = texture(src, vec2(tex_coord.x + 2*x, tex_coord.y - 2*y)).rgb;

                        vec3 j = texture(src, vec2(tex_coord.x - x, tex_coord.y + y)).rgb;
                        vec3 k = texture(src, vec2(tex_coord.x + x, tex_coord.y + y)).rgb;
                        vec3 l = texture(src, vec2(tex_coord.x - x, tex_coord.y - y)).rgb;
                        vec3 m = texture(src, vec2(tex_coord.x + x, tex_coord.y - y)).rgb;

                        // Weighted distribution:
                        // 0.5 + 0.125 + 0.125 + 0.125 + 0.125 = 1
                        // Center box (j,k,l,m) = 0.5
                        // Corner boxes = 0.125 each
                        vec3 result = e * 0.125;
                        result += (a + c + g + i) * 0.03125;
                        result += (b + d + f + h) * 0.0625;
                        result += (j + k + l + m) * 0.125;

                        imageStore(dst, uv, vec4(result, 1.0));
                    }
                "
            }
        }

        // 3x3 tent filter upsample (bilinear + additive)
        mod cs_upsample {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 450
                    layout(local_size_x = 8, local_size_y = 8) in;
                    layout(set=0, binding=0) uniform sampler2D src;
                    layout(set=0, binding=1, rgba16f) uniform image2D dst;

                    layout(push_constant) uniform Push {
                        float filter_radius;
                        float intensity;
                    } pc;

                    void main() {
                        ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
                        ivec2 size = imageSize(dst);
                        if (uv.x >= size.x || uv.y >= size.y) return;

                        // Texture coordinate in [0,1]
                        vec2 tex_coord = (vec2(uv) + 0.5) / vec2(size);

                        float x = pc.filter_radius;
                        float y = pc.filter_radius;

                        // 3x3 tent filter
                        // a - b - c
                        // d - e - f
                        // g - h - i
                        vec3 a = texture(src, vec2(tex_coord.x - x, tex_coord.y + y)).rgb;
                        vec3 b = texture(src, vec2(tex_coord.x,     tex_coord.y + y)).rgb;
                        vec3 c = texture(src, vec2(tex_coord.x + x, tex_coord.y + y)).rgb;

                        vec3 d = texture(src, vec2(tex_coord.x - x, tex_coord.y)).rgb;
                        vec3 e = texture(src, vec2(tex_coord.x,     tex_coord.y)).rgb;
                        vec3 f = texture(src, vec2(tex_coord.x + x, tex_coord.y)).rgb;

                        vec3 g = texture(src, vec2(tex_coord.x - x, tex_coord.y - y)).rgb;
                        vec3 h = texture(src, vec2(tex_coord.x,     tex_coord.y - y)).rgb;
                        vec3 i = texture(src, vec2(tex_coord.x + x, tex_coord.y - y)).rgb;

                        // 3x3 tent filter weights:
                        // 1/16 * | 1 2 1 |
                        //        | 2 4 2 |
                        //        | 1 2 1 |
                        vec3 upsample = e * 4.0;
                        upsample += (b + d + f + h) * 2.0;
                        upsample += (a + c + g + i);
                        upsample *= 1.0 / 16.0;

                        // Additive blend with destination
                        vec4 existing = imageLoad(dst, uv);
                        imageStore(dst, uv, vec4(existing.rgb + upsample * pc.intensity, 1.0));
                    }
                "
            }
        }

        // Build extract stage
        let (extract_layout, extract_pipe_layout) =
            make_storage_layout(device, std::mem::size_of::<f32>() as u32);
        let extract_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(
                    cs_extract::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                ),
                extract_pipe_layout,
            ),
        )
        .unwrap();
        let extract = BloomStage {
            pipeline: extract_pipeline,
            layout: extract_layout,
        };

        // Build downsample stage (vec2 push constant for src resolution)
        let (downsample_layout, downsample_pipe_layout) =
            make_sampler_layout(device, std::mem::size_of::<[f32; 2]>() as u32);
        let downsample_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(
                    cs_downsample::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                ),
                downsample_pipe_layout,
            ),
        )
        .unwrap();
        let downsample = BloomStage {
            pipeline: downsample_pipeline,
            layout: downsample_layout,
        };

        // Build upsample stage (filter_radius + intensity)
        let (upsample_layout, upsample_pipe_layout) =
            make_sampler_layout(device, std::mem::size_of::<[f32; 2]>() as u32);
        let upsample_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(
                PipelineShaderStageCreateInfo::new(
                    cs_upsample::load(device.clone())
                        .unwrap()
                        .entry_point("main")
                        .unwrap(),
                ),
                upsample_pipe_layout,
            ),
        )
        .unwrap();
        let upsample = BloomStage {
            pipeline: upsample_pipeline,
            layout: upsample_layout,
        };

        Ok(Self {
            extract,
            downsample,
            upsample,
            sampler,
            mip_views,
            mip_sizes,
            mip_count,
        })
    }

    pub fn run(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        descriptor_allocator: &Arc<StandardDescriptorSetAllocator>,
        hdr_input: Arc<ImageView>,
        settings: &BloomSettings,
    ) {
        let (w0, h0) = self.mip_sizes[0];

        let threshold = settings
            .threshold
            .clamp(bloom_limits::THRESHOLD_MIN, bloom_limits::THRESHOLD_MAX);

        let intensity = settings
            .intensity
            .clamp(bloom_limits::INTENSITY_MIN, bloom_limits::INTENSITY_MAX);

        let filter_radius = settings.filter_radius.clamp(
            bloom_limits::FILTER_RADIUS_MIN,
            bloom_limits::FILTER_RADIUS_MAX,
        );

        builder
            .begin_debug_utils_label(DebugUtilsLabel {
                label_name: "Bloom".to_string(),
                color: [0.2, 0.3, 0.1, 1.0],
                ..Default::default()
            })
            .unwrap();

        // ===== EXTRACT =====
        let extract_set = DescriptorSet::new(
            descriptor_allocator.clone(),
            self.extract.layout.clone(),
            [
                WriteDescriptorSet::image_view(0, hdr_input.clone()),
                WriteDescriptorSet::image_view(1, self.mip_views[0].clone()),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.extract.pipeline.clone())
            .unwrap()
            .push_constants(self.extract.pipeline.layout().clone(), 0, threshold)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.extract.pipeline.layout().clone(),
                0,
                extract_set,
            )
            .unwrap()
            .begin_debug_utils_label(DebugUtilsLabel {
                label_name: "Extraction".to_string(),
                color: [0.9, 0.1, 0.1, 1.0],
                ..Default::default()
            })
            .unwrap();

        unsafe {
            builder
                .dispatch([w0.div_ceil(8), h0.div_ceil(8), 1])
                .unwrap()
                .end_debug_utils_label()
                .unwrap();
        }

        // ===== DOWNSAMPLE CHAIN =====
        for mip in 1..self.mip_count {
            let (src_w, src_h) = self.mip_sizes[mip - 1];
            let (dst_w, dst_h) = self.mip_sizes[mip];

            let down_set = DescriptorSet::new(
                descriptor_allocator.clone(),
                self.downsample.layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        self.mip_views[mip - 1].clone(),
                        self.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view(1, self.mip_views[mip].clone()),
                ],
                [],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(self.downsample.pipeline.clone())
                .unwrap()
                .push_constants(
                    self.downsample.pipeline.layout().clone(),
                    0,
                    [src_w as f32, src_h as f32],
                )
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.downsample.pipeline.layout().clone(),
                    0,
                    down_set,
                )
                .unwrap()
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: format!("Downsample({}->{})", mip - 1, mip),
                    color: [0.1, 0.2, 0.9, 1.0],
                    ..Default::default()
                })
                .unwrap();

            unsafe {
                builder
                    .dispatch([dst_w.div_ceil(8), dst_h.div_ceil(8), 1])
                    .unwrap()
                    .end_debug_utils_label()
                    .unwrap();
            }
        }

        // ===== UPSAMPLE CHAIN =====
        for mip in (1..self.mip_count).rev() {
            let (dst_w, dst_h) = self.mip_sizes[mip - 1];

            let up_set = DescriptorSet::new(
                descriptor_allocator.clone(),
                self.upsample.layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        self.mip_views[mip].clone(),
                        self.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view(1, self.mip_views[mip - 1].clone()),
                ],
                [],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(self.upsample.pipeline.clone())
                .unwrap()
                .push_constants(
                    self.upsample.pipeline.layout().clone(),
                    0,
                    [filter_radius, intensity],
                )
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.upsample.pipeline.layout().clone(),
                    0,
                    up_set,
                )
                .unwrap()
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: format!("Upsample({}->{})", mip, mip - 1),
                    color: [0.8, 0.1, 0.8, 1.0],
                    ..Default::default()
                })
                .unwrap();

            unsafe {
                builder
                    .dispatch([dst_w.div_ceil(8), dst_h.div_ceil(8), 1])
                    .unwrap()
                    .end_debug_utils_label()
                    .unwrap();
            }
        }

        unsafe {
            builder.end_debug_utils_label().unwrap();
        }
    }

    pub fn result(&self) -> Arc<ImageView> {
        self.mip_views[0].clone()
    }
}
