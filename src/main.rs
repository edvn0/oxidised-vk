mod bloom_pass;
mod camera;
mod input_state;
mod main_helpers;
mod math;
mod shader_bindings;
mod culling;

use crate::bloom_pass::BloomPass;
use crate::camera::Camera;
use crate::input_state::InputState;
use crate::main_helpers::FrameDescriptorSet;
use crate::shader_bindings::{RendererUBO, renderer_set_0_layouts};
use nalgebra::{Matrix4, Rotation3, Translation3, Vector3};
use std::collections::BTreeMap;
use std::default::Default;
use std::time::Instant;
use std::{error::Error, sync::Arc};
use rand::Rng;
use vulkano::command_buffer::{ClearColorImageInfo, DrawIndexedIndirectCommand, PrimaryCommandBufferAbstract};
use vulkano::descriptor_set::layout::DescriptorType::{CombinedImageSampler, StorageBuffer};
use vulkano::format::{ClearColorValue, ClearValue, FormatFeatures};
use vulkano::image::sampler::{
    BorderColor, Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode, SamplerReductionMode,
};
use vulkano::image::view::{ImageViewCreateInfo, ImageViewType};
use vulkano::image::{ImageAspects, ImageSubresourceRange, SampleCount};
use vulkano::instance::InstanceExtensions;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::vertex_input::VertexInputState;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::{
    DeviceSize, Validated, Version, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    shader::ShaderStages,
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::ComputePipeline;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use crate::culling::{construct_culling_passes, CullPass, PrefixPass};

extern crate nalgebra_glm as glm;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop).unwrap();

    event_loop.run_app(&mut app)
}

#[repr(C)]
#[derive(BufferContents)]
struct PushConsts {
    model: [f32; 16],
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    index_buffer: Subbuffer<[u32]>,
    rcx: Option<RenderContext>,

    input_state: InputState,
    camera: Camera,
    last_frame: Instant,
}
struct MRT {
    predepth_pipeline: Arc<GraphicsPipeline>,
    gbuffer_image_views: [Arc<ImageView>; 3],
    gbuffer_depth_view: Arc<ImageView>,
    gbuffer_pipeline: Arc<GraphicsPipeline>,
    gbuffer_instanced_pipeline: Arc<GraphicsPipeline>,
}

struct SwapchainPass {
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    attachment_image_views: Vec<Arc<ImageView>>,
}

struct MRTLighting {
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    image_view: Arc<ImageView>,
}

struct Composite {
    pipeline: Arc<GraphicsPipeline>,
    set: Arc<DescriptorSet>,
    image_view: Arc<ImageView>,
}

#[repr(C)]
#[derive(BufferContents, Copy, Clone)]
pub struct TransformTRS {
    pub trs: [f32; 16],
}

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct FrustumPlanes {
    pub planes: [[f32; 4]; 6],
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    viewport: Viewport,
    scissor: Scissor,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    default_sampler: Arc<Sampler>,
    start_time: Instant,
    elapsed_millis: u64,

    frame_transforms: Vec<Subbuffer<[TransformTRS]>>, // TODO: Move
    indirect_buffers: Vec<Subbuffer<[DrawIndexedIndirectCommand]>>,
    visibility_buffers: Vec<Subbuffer<[u32]>>,
    mesh_offsets_buffers: Vec<Subbuffer<[u32]>>,
    white_image_sampler: Arc<ImageView>,
    black_image_sampler: Arc<ImageView>,

    context_descriptor_set: FrameDescriptorSet,
    frame_uniform_buffers: Vec<Subbuffer<RendererUBO>>,
    frustum_buffers: Vec<Subbuffer<FrustumPlanes>>,

    mrt_pass: MRT,
    mrt_lighting: MRTLighting,
    composite: Composite,
    swapchain_pass: SwapchainPass,
    bloom_pass: BloomPass,
    cull_pass: CullPass,
    prefix_pass: PrefixPass
}

impl RenderContext {
    pub fn update_camera_ubo(
        &self,
        camera: &Camera,
        image_index: usize,
        window_size: PhysicalSize<u32>,
    ) {
        let aspect = window_size.width as f32 / window_size.height as f32;

        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let inverse_proj = proj.try_inverse().unwrap();

        let sun = camera.sun_direction_view_space();

        if let Ok(mut w) = self.frame_uniform_buffers[image_index].write() {
            *w = RendererUBO {
                view: view.as_slice().try_into().unwrap(),
                proj: proj.as_slice().try_into().unwrap(),
                inverse_proj: inverse_proj.as_slice().try_into().unwrap(),
                sun_direction: sun,
            };
        }
    }
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Result<Self, Box<dyn Error>> {
        let library = VulkanLibrary::new().unwrap();

        let enabled_extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..Surface::required_extensions(event_loop).unwrap()
        };
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, graphics_index, compute_index) = {
            let mut best = None;
            for p in instance.enumerate_physical_devices().unwrap() {
                if !(p.api_version() >= Version::V1_4 || p.supported_extensions().khr_dynamic_rendering) {
                    continue;
                }
                if !p.supported_extensions().contains(&device_extensions) {
                    continue;
                }

                let mut gq = None;
                let mut cq = None;

                for (i, q) in p.queue_family_properties().iter().enumerate() {
                    if gq.is_none()
                        && q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.presentation_support(i as u32, event_loop).unwrap()
                    {
                        gq = Some(i as u32);
                    }
                    if cq.is_none() && q.queue_flags.intersects(QueueFlags::COMPUTE) {
                        cq = Some(i as u32);
                    }
                }

                let g = match gq { Some(v) => v, None => continue };
                let c = cq.unwrap_or(g);

                let score = match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    _ => 4,
                };

                match best {
                    None => best = Some((p, g, c, score)),
                    Some((_, _, _, s)) if score < s => best = Some((p, g, c, score)),
                    _ => {}
                }
            }
            let (p, g, c, _) = best.expect("no suitable device found");
            (p, g, c)
        };

        let mut infos = Vec::new();
        infos.push(QueueCreateInfo { queue_family_index: graphics_index, ..Default::default() });
        if compute_index != graphics_index {
            infos.push(QueueCreateInfo { queue_family_index: compute_index, ..Default::default() });
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: infos,
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures { dynamic_rendering: true, ..DeviceFeatures::empty() },
                ..Default::default()
            },
        ).unwrap();

        let graphics_queue = queues.next().unwrap();
        let compute_queue = if compute_index != graphics_index {
            queues.next().unwrap()
        } else {
            graphics_queue.clone()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let (models, _materials) = tobj::load_obj(
            "assets/torus_knot.obj",
            &tobj::LoadOptions {
                triangulate: true,
                single_index: true,
                ..Default::default()
            },
        )?;

        let mesh = &models[0].mesh;

        let mut vertices = Vec::with_capacity(mesh.indices.len());
        let mut indices = Vec::with_capacity(mesh.indices.len());

        for face_index in mesh.indices.iter() {
            let i = *face_index as usize;

            let px = mesh.positions[i * 3 + 0];
            let py = mesh.positions[i * 3 + 1];
            let pz = mesh.positions[i * 3 + 2];

            let (nx, ny, nz) = if !mesh.normals.is_empty() {
                (
                    mesh.normals[i * 3 + 0],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                )
            } else {
                (0.0, 0.0, 0.0)
            };

            let (ux, uy) = if !mesh.texcoords.is_empty() {
                (mesh.texcoords[i * 2 + 0], mesh.texcoords[i * 2 + 1])
            } else {
                (0.0, 0.0)
            };

            indices.push(vertices.len() as u32);

            vertices.push(MyVertex {
                position: [px, py, pz],
                normal: [nx, ny, nz],
                uvs: [ux, uy],
            });
        }

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let index_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        Ok(App {
            instance,
            device,
            graphics_queue,
            compute_queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            vertex_buffer,
            index_buffer,
            input_state: InputState::new(),
            camera: Camera::new(),
            last_frame: Instant::now(),
            rcx: None,
        })
    }
}

fn generate_random_transforms(count: usize) -> Vec<TransformTRS> {
    let mut rng = rand::rng();
    let mut transforms = Vec::with_capacity(count);

    for _ in 0..count {
        let x = rng.random_range(-50.0..50.0);
        let y = rng.random_range(-50.0..50.0);
        let z = rng.random_range(-50.0..50.0);
        let translation = Translation3::new(x, y, z);

        let rx = rng.random_range(0.0..std::f32::consts::TAU);
        let ry = rng.random_range(0.0..std::f32::consts::TAU);
        let rz = rng.random_range(0.0..std::f32::consts::TAU);
        let rotation = nalgebra::Rotation3::from_euler_angles(rx, ry, rz);

        let scale = rng.random_range(0.1..0.5);
        let scale_matrix = Matrix4::new_scaling(scale);

        let transform = translation.to_homogeneous() * rotation.to_homogeneous() * scale_matrix;

        transforms.push(TransformTRS {
            trs: transform.as_slice().try_into().unwrap(),
        });
    }

    transforms
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut attribs = Window::default_attributes();
        attribs.inner_size = Some(PhysicalSize::new(1920, 1280).into());
        let window = Arc::new(event_loop.create_window(attribs).unwrap());
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, swapchain_images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let formats = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap();

            let image_format = formats
                .iter()
                .find(|(fmt, _)| fmt == &Format::B8G8R8A8_SRGB || fmt == &Format::R8G8B8A8_SRGB)
                .map(|(fmt, _)| *fmt)
                .unwrap_or(formats[0].0); // fallback if no srgb is available

            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),

                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let default_sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Nearest,
                min_filter: Filter::Nearest,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::Repeat,
                ],
                mip_lod_bias: 0.0,
                anisotropy: None,
                compare: None,
                lod: 0.0..=1.0,
                border_color: BorderColor::FloatTransparentBlack,
                unnormalized_coordinates: false,
                reduction_mode: SamplerReductionMode::WeightedAverage,
                sampler_ycbcr_conversion: None,
                ..Default::default()
            },
        )
        .unwrap();

        let (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass, cull, prefix) =
            window_size_dependent_setup(
                &self.device,
                &self.memory_allocator,
                &self.descriptor_set_allocator,
                &default_sampler,
                &swapchain_images,
            );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [window_size.width as f32, window_size.height as f32],
            depth_range: 1.0..=0.0, // Use standard range with reverse-Z in shader
        };

        let scissor = Scissor {
            offset: [0, 0],
            extent: window_size.into(),
        };

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        let image_count = swapchain.image_count();

        let uniform_buffers = (0..image_count)
            .map(|_| {
                Buffer::new_sized::<RendererUBO>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<Subbuffer<RendererUBO>>>();

        let frustum_ubos = (0..image_count)
            .map(|_| {
                Buffer::new_sized::<FrustumPlanes>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                ).unwrap()
            })
            .collect::<Vec<Subbuffer<FrustumPlanes>>>();

        let ssbos = (0..image_count)
            .map(|_| {
                Buffer::new_slice::<TransformTRS>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    10_000 * std::mem::size_of::<TransformTRS>() as DeviceSize,
                )
                .unwrap()
            })
            .collect::<Vec<Subbuffer<[TransformTRS]>>>();

        let indirect_buffers = (0..image_count)
            .map(|_| {
                Buffer::new_slice::<DrawIndexedIndirectCommand>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    1
                ).unwrap()
            })
            .collect::<Vec<_>>();

        let visibility_buffers: Vec<_> = (0..image_count)
            .map(|_| {
                Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    std::iter::repeat(0u32).take(10_000),
                ).unwrap()
            }).collect();

        let mesh_offsets_buffers: Vec<_> = (0..image_count)
            .map(|_| {
                Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                        ..Default::default()
                    },
                    [0u32, 10_000u32],
                ).unwrap()
            }).collect();

        let _shadow_map_sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [
                    SamplerAddressMode::ClampToBorder,
                    SamplerAddressMode::ClampToBorder,
                    SamplerAddressMode::ClampToBorder,
                ],
                compare: Some(CompareOp::LessOrEqual),
                border_color: BorderColor::FloatOpaqueWhite,
                lod: 0.0..=LOD_CLAMP_NONE,
                unnormalized_coordinates: false,
                ..Default::default()
            },
        )
        .unwrap();

        let random_transforms = generate_random_transforms(100_000);

        for ssbo in ssbos.iter() {
            if let Ok(mut write) = ssbo.write() {
                let len = write.len().min(random_transforms.len());
                write[..len].copy_from_slice(&random_transforms[..len]);
            }
        }

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            swapchain_pass,
            viewport,
            scissor,
            recreate_swapchain,
            previous_frame_end,
            start_time: Instant::now(),
            elapsed_millis: 0,
            context_descriptor_set: FrameDescriptorSet::new(
                self.device.clone(),
                self.descriptor_set_allocator.clone(),
                &uniform_buffers,
            ),
            default_sampler,
            frame_transforms: ssbos,
            indirect_buffers,
            visibility_buffers,
            mesh_offsets_buffers,
            white_image_sampler: create_image(
                &self.graphics_queue,
                &self.command_buffer_allocator,
                &self.memory_allocator,
                0xFF,
            ),
            black_image_sampler: create_image(
                &self.graphics_queue,
                &self.command_buffer_allocator,
                &self.memory_allocator,
                0x00,
            ),
            frame_uniform_buffers: uniform_buffers,
            frustum_buffers: frustum_ubos,
            mrt_pass: mrt,
            mrt_lighting,
            composite,
            bloom_pass,
            cull_pass: cull,
            prefix_pass: prefix,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { .. } => {
                if let WindowEvent::KeyboardInput { event, .. } = &event
                    && event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                    && event.state == ElementState::Pressed
                    && !event.repeat
                {
                    event_loop.exit();
                    return;
                }
                self.input_state.process_input(&event);
                self.input_state
                    .apply_cursor_mode(self.rcx.as_ref().unwrap().window.as_ref());
            }

            WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. } => {
                self.input_state.process_input(&event);
                self.input_state
                    .apply_cursor_mode(self.rcx.as_ref().unwrap().window.as_ref());
            }

            WindowEvent::Resized(_) => {
                self.rcx.as_mut().unwrap().recreate_swapchain = true;
            }

            WindowEvent::RedrawRequested => self.render_frame(),

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event
            && self.input_state.rotating
        {
            self.input_state.mouse_delta = (delta.0 as f32, delta.1 as f32);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

fn create_image(
    queue: &Arc<Queue>,
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    allocator: &Arc<StandardMemoryAllocator>,
    value: i32,
) -> Arc<ImageView> {
    let img = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [1, 1, 1],
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Sample1,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            initial_layout: vulkano::image::ImageLayout::Undefined,
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();

    {
        // Perform a clear to the specified value
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let val_as_f32 = if value > 1 {
            value as f32 / 255.0
        } else {
            value as f32
        };

        let mut clear_info = ClearColorImageInfo::image(img.clone());
        clear_info.clear_value =
            ClearColorValue::Float([val_as_f32, val_as_f32, val_as_f32, 1.0f32]);

        builder.clear_color_image(clear_info).unwrap();

        let result = builder.build().unwrap().execute(queue.clone()).unwrap();
        result
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    ImageView::new_default(img.clone()).unwrap()
}

impl App {
    fn render_frame(&mut self) {
        let rcx = self.rcx.as_mut().unwrap();

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.camera.update_from_input(&self.input_state, dt);

        let window_size = rcx.window.inner_size();
        rcx.elapsed_millis = rcx.start_time.elapsed().as_millis() as u64;

        // Do not draw the frame when the screen size is zero. On Windows, this can occur
        // when minimizing the application.
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        if rcx.recreate_swapchain {
            let (new_swapchain, new_swapchain_images) = rcx
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            rcx.swapchain = new_swapchain;

            let (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass, cull, prefix) =
                window_size_dependent_setup(
                    &self.device,
                    &self.memory_allocator,
                    &self.descriptor_set_allocator,
                    &rcx.default_sampler,
                    &new_swapchain_images,
                );
            rcx.swapchain_pass = swapchain_pass;
            rcx.mrt_pass = mrt;
            rcx.mrt_lighting = mrt_lighting;
            rcx.composite = composite;
            rcx.bloom_pass = bloom_pass;
            rcx.cull_pass= cull;
            rcx.prefix_pass= prefix;

            rcx.viewport.extent = [window_size.width as f32, window_size.height as f32];
            rcx.scissor.extent = window_size.into();

            rcx.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        {
            let ib = &rcx.indirect_buffers[image_index as usize];
            if let Ok(mut d) = ib.write() {
                for k in d.iter_mut() {
                    k.index_count = self.index_buffer.len() as u32;
                    k.instance_count = 0;
                    k.first_index = 0;
                    k.vertex_offset = 0;
                    k.first_instance = 0;
                }
            }
        }


        {
            let view = self.camera.view_matrix();
            let proj = self.camera.projection_matrix(
                rcx.window.inner_size().width as f32
                    / rcx.window.inner_size().height as f32
            );
            let vp = proj * view;

            let mut planes = [[0f32; 4]; 6];

            let m = vp.as_slice();

            fn extract_plane(m: &[f32], a: usize, sign: f32) -> [f32; 4] {
                [
                    m[12] + sign * m[a + 0], // Nx
                    m[13] + sign * m[a + 4], // Ny
                    m[14] + sign * m[a + 8], // Nz
                    m[15] + sign * m[a + 12] // D
                ]
            }

            planes[0] = extract_plane(m, 0,  1.0); // Left
            planes[1] = extract_plane(m, 0, -1.0); // Right
            planes[2] = extract_plane(m, 1,  1.0); // Bottom
            planes[3] = extract_plane(m, 1, -1.0); // Top
            planes[4] = extract_plane(m, 2,  1.0); // Near
            planes[5] = extract_plane(m, 2, -1.0); // Far

            fn normalize_plane(p: &mut [f32; 4]) {
                let len = (p[0]*p[0] + p[1]*p[1] + p[2]*p[2]).sqrt();
                p[0] /= len;
                p[1] /= len;
                p[2] /= len;
                p[3] /= len;
            }

            for p in planes.iter_mut() {
                normalize_plane(p);
            }

            if let Ok(mut w) = rcx.frustum_buffers[image_index as usize].write() {
                w.planes = planes;
            }
        }


        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        acquire_future.wait(None).unwrap();
        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        {
            rcx.update_camera_ubo(&self.camera, image_index as usize, window_size);
        }

        let culling_future = {
            let transforms = rcx.frame_transforms[image_index as usize].clone();
            let mesh_count = 1;

            let visibility = rcx.visibility_buffers[image_index as usize].clone();
            if let Ok(mut v) = visibility.write() {
                for x in v.iter_mut() { *x = 0; }
            }
            let mesh_offsets = rcx.mesh_offsets_buffers[image_index as usize].clone();

            // Pass 1: Culling
            let fut_cull = rcx.cull_pass.execute(
                &self.command_buffer_allocator,
                &self.compute_queue,
                &self.descriptor_set_allocator,
                transforms.clone(),
                visibility.clone(),
                rcx.frustum_buffers[image_index as usize].clone(),
            );

            // Pass 2: Prefix sum + indirect write
            let fut_prefix = rcx.prefix_pass.execute(
                &self.command_buffer_allocator,
                &self.compute_queue,
                &self.descriptor_set_allocator,
                visibility.clone(),
                rcx.indirect_buffers[image_index as usize].clone(),
                mesh_offsets.clone(),
                mesh_count,
            );

            fut_cull.join(fut_prefix)
                .then_signal_semaphore_and_flush()
                .unwrap()
        };

        {
            // Pre-Depth instanced pass
            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                rcx.mrt_pass.predepth_pipeline.layout().set_layouts()[1].clone(),
                [WriteDescriptorSet::buffer(
                    1,
                    rcx.frame_transforms[image_index as usize].clone(),
                )],
                [],
            ).unwrap();

            let sets = [
                rcx.context_descriptor_set.for_frame(image_index as usize).clone(),
                set,
            ];

            builder
                .begin_rendering(RenderingInfo {
                    render_area_extent: rcx.scissor.extent,
                    render_area_offset: rcx.scissor.offset,
                    color_attachments: vec![], // No MRT!
                    depth_attachment: Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some(ClearValue::Depth(0.0)),
                        ..RenderingAttachmentInfo::image_view(
                            rcx.mrt_pass.gbuffer_depth_view.clone(),
                        )
                    }),
                    ..Default::default()
                }).unwrap()
                .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                .unwrap()
                .set_scissor(0, [rcx.scissor].into_iter().collect())
                .unwrap()
                .bind_pipeline_graphics(rcx.mrt_pass.predepth_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    rcx.mrt_pass.predepth_pipeline.layout().clone(),
                    0,
                    sets.to_vec(),
                )
                .unwrap()
                .bind_vertex_buffers(0, self.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(self.index_buffer.clone())
                .unwrap();

            unsafe {
                builder.draw_indexed_indirect(
                    rcx.indirect_buffers[image_index as usize].clone(),
                ).unwrap();
            }
            builder.end_rendering().unwrap();
        }

        {
            // Pass 1: MRT with instanced geometry

            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                rcx.mrt_pass.gbuffer_instanced_pipeline.layout().set_layouts()[1].clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    rcx.white_image_sampler.clone(),
                    rcx.default_sampler.clone(),
                ),
                    WriteDescriptorSet::buffer(
                        1,
                        rcx.frame_transforms[image_index as usize].clone(),
                    )],
                [],
            )
                .unwrap();

            let sets = [
                rcx.context_descriptor_set
                    .for_frame(image_index as usize)
                    .clone(),
                set,
            ];

            builder
                .begin_rendering(RenderingInfo {
                    render_area_extent: rcx.scissor.extent,
                    render_area_offset: rcx.scissor.offset,
                    color_attachments: vec![
                        Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                            ..RenderingAttachmentInfo::image_view(
                                rcx.mrt_pass.gbuffer_image_views[0].clone(),
                            )
                        }),
                        Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                            ..RenderingAttachmentInfo::image_view(
                                rcx.mrt_pass.gbuffer_image_views[1].clone(),
                            )
                        }),
                        Some(RenderingAttachmentInfo {
                            load_op: AttachmentLoadOp::Clear,
                            store_op: AttachmentStoreOp::Store,
                            clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                            ..RenderingAttachmentInfo::image_view(
                                rcx.mrt_pass.gbuffer_image_views[2].clone(),
                            )
                        }),
                    ],
                    depth_attachment: Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Load,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: None,
                        ..RenderingAttachmentInfo::image_view(
                            rcx.mrt_pass.gbuffer_depth_view.clone(),
                        )
                    }),
                    ..Default::default()
                })
                .unwrap()
                .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                .unwrap()
                .set_scissor(0, [rcx.scissor].into_iter().collect())
                .unwrap()
                .bind_pipeline_graphics(rcx.mrt_pass.gbuffer_instanced_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    rcx.mrt_pass.gbuffer_instanced_pipeline.layout().clone(),
                    0,
                    sets.to_vec(),
                )
                .unwrap()
                .bind_vertex_buffers(0, self.vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(self.index_buffer.clone())
                .unwrap();

            unsafe {
                builder.draw_indexed_indirect(
                    rcx.indirect_buffers[image_index as usize].clone(),
                ).unwrap();
            }
            builder.end_rendering().unwrap();
        }

        {
            // Pass 2: MRT Lighting
            let descriptor_sets = vec![
                rcx.context_descriptor_set
                    .for_frame(image_index as usize)
                    .clone(),
                rcx.mrt_lighting.set.clone(),
            ];

            builder
                .begin_rendering(RenderingInfo {
                    render_area_extent: rcx.scissor.extent,
                    render_area_offset: rcx.scissor.offset,
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(rcx.mrt_lighting.image_view.clone())
                    })],
                    ..Default::default()
                })
                .unwrap()
                .bind_pipeline_graphics(rcx.mrt_lighting.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    rcx.mrt_lighting.pipeline.layout().clone(),
                    0,
                    descriptor_sets,
                )
                .unwrap();

            unsafe { builder.draw(3, 1, 0, 0).unwrap() };
            builder.end_rendering().unwrap();
        }

        {
            // Pass 2.1: Bloom
            rcx.bloom_pass.run(
                &mut builder,
                &self.descriptor_set_allocator,
                rcx.mrt_lighting.image_view.clone(),
                1.0, // bloom intensity
                1.0, // threshold
            );
        }

        {
            // Pass 3: Compositing, HDR -> HDR
            let descriptor_sets = vec![rcx.composite.set.clone()];

            const PCS: [f32; 1] = [1.0];

            builder
                .begin_rendering(RenderingInfo {
                    render_area_extent: rcx.scissor.extent,
                    render_area_offset: rcx.scissor.offset,
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(rcx.composite.image_view.clone())
                    })],
                    ..Default::default()
                })
                .unwrap()
                .bind_pipeline_graphics(rcx.composite.pipeline.clone())
                .unwrap()
                .push_constants(rcx.composite.pipeline.layout().clone(), 0, PCS)
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    rcx.composite.pipeline.layout().clone(),
                    0,
                    descriptor_sets,
                )
                .unwrap();

            unsafe { builder.draw(3, 1, 0, 0).unwrap() };
            builder.end_rendering().unwrap();
        }

        {
            // Pass 4: Present, HDR -> LDR with tonemapping

            let descriptor_sets = vec![rcx.swapchain_pass.set.clone()];

            builder
                .begin_rendering(RenderingInfo {
                    render_area_extent: rcx.scissor.extent,
                    render_area_offset: rcx.scissor.offset,
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(
                            rcx.swapchain_pass.attachment_image_views[image_index as usize].clone(),
                        )
                    })],
                    ..Default::default()
                })
                .unwrap()
                .bind_pipeline_graphics(rcx.swapchain_pass.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    rcx.swapchain_pass.pipeline.layout().clone(),
                    0,
                    descriptor_sets,
                )
                .unwrap();

            unsafe { builder.draw(3, 1, 0, 0).unwrap() };
            builder.end_rendering().unwrap();
        }

        let command_buffer = builder.build().unwrap();

        let future = rcx
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .join(culling_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }

        self.input_state.end_frame();
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    uvs: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
}

fn window_size_dependent_setup(
    device: &Arc<Device>,
    memory_allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    default_sampler: &Arc<Sampler>,
    swapchain_images: &[Arc<Image>],
) -> (SwapchainPass, MRT, MRTLighting, Composite, BloomPass, CullPass, PrefixPass) {
    let predepth_pipeline = {
        mod vs {
            vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450
                layout(location = 0) in vec3 position;

                layout(set=0, binding=0, std140) uniform UBO {
                    mat4 view;
                    mat4 projection;
                    mat4 inverse_projection;
                    vec4 sun_direction;
                };

                struct Transform { mat4 t; };
                layout(set = 1, binding = 1, std140) readonly buffer SSBO {
                    Transform transforms[];
                };

                void main() {
                    int index = gl_InstanceIndex;
                    mat4 model = transforms[index].t;

                    vec4 world_pos = model * vec4(position,1.0);
                    gl_Position = projection * (view * world_pos);
                }
            "
        }
        }

        mod fs {
            vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450
                void main() { }
            "
        }
        }

        let vs = vs::load(device.clone()).unwrap().entry_point("main").unwrap();
        let fs = fs::load(device.clone()).unwrap().entry_point("main").unwrap();

        let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let subpass = PipelineRenderingCreateInfo {
            depth_attachment_format: Some(Format::D24_UNORM_S8_UINT),
            ..Default::default()
        };

        let layout = {
            let mut set_layouts = renderer_set_0_layouts();
            // Keep instancing same as MRT
            let mut inst = DescriptorSetLayoutCreateInfo::default();
            inst.bindings.insert(1, {
                let mut b = DescriptorSetLayoutBinding::descriptor_type(StorageBuffer);
                b.stages = ShaderStages::VERTEX;
                b
            });
            set_layouts.push(inst);
            PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo {
                    flags: Default::default(),
                    set_layouts,
                    push_constant_ranges: vec![],
                }
                    .into_pipeline_layout_create_info(device.clone()).unwrap(),
            )
                .unwrap()
        };

        let mut ci = GraphicsPipelineCreateInfo::layout(layout);
        ci.vertex_input_state = Some(vertex_input_state);
        ci.input_assembly_state = Some(InputAssemblyState::default());
        ci.viewport_state = Some(ViewportState::default());
        ci.rasterization_state = Some(RasterizationState::default());
        ci.multisample_state = Some(MultisampleState::default());
        ci.depth_stencil_state = Some(DepthStencilState {
            depth: Some(DepthState::reverse()), // Important
            ..Default::default()
        });
        ci.subpass = Some(subpass.into());
        ci.stages = stages.into_iter().collect();
        ci.dynamic_state.insert(DynamicState::Viewport);
        ci.dynamic_state.insert(DynamicState::Scissor);

        GraphicsPipeline::new(device.clone(), None, ci).unwrap()
    };

    // ==============================================================
    // === MRT GEOMETRY PASS (Unmodified from your version)
    // ==============================================================
    let mrt_pipeline = {
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 position;
                    layout(location = 1) in vec2 uvs;
                    layout(location = 2) in vec3 normal;

                    layout(location = 0) out vec3 v_normal;
                    layout(location = 1) out vec2 v_uvs;

                    layout(set=0, binding=0, std140) uniform UBO {
                        mat4 view;
                        mat4 projection;
                        mat4 inverse_projection;
                        vec4 sun_direction;
                    };

                    layout(push_constant, std430) uniform PC {
                        mat4 model;
                    };

                    void main() {
                        mat3 normal_mat = transpose(inverse(mat3(model)));
                        vec3 world_norm = normalize(normal_mat * normal);

                        vec4 world_pos = model * vec4(position,1.0);
                        vec4 view_pos = view * world_pos;

                        v_normal = normalize((view * vec4(world_norm,0.0)).xyz);
                        v_uvs = uvs;

                        gl_Position = projection * view_pos;
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 v_normal;
                    layout(location = 1) in vec2 v_uvs;

                    layout(location = 0) out vec4 normal_mrt;
                    layout(location = 1) out vec4 uvs_mrt;
                    layout(location = 2) out vec4 albedo_mrt;

                    layout (set = 1, binding = 0) uniform sampler2D albedo_tex;

                    void main() {
                        normal_mrt = vec4(normalize(v_normal), 0.0);
                        uvs_mrt = vec4(v_uvs, 0.0, 0.0);
                        albedo_mrt = texture(albedo_tex, v_uvs);
                    }
                ",
            }
        }

        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![
                Some(Format::R16G16B16A16_SFLOAT),
                Some(Format::R16G16B16A16_SFLOAT),
                Some(Format::R16G16B16A16_SFLOAT),
            ],
            depth_attachment_format: Some(Format::D24_UNORM_S8_UINT),
            ..Default::default()
        };

        const PC_RANGE: PushConstantRange = PushConstantRange {
            stages: ShaderStages::VERTEX,
            offset: 0,
            size: std::mem::size_of::<PushConsts>() as u32,
        };

        let layout = {
            let mut set_layout_1 = DescriptorSetLayoutCreateInfo::default();
            let mut bind = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
            bind.stages = ShaderStages::FRAGMENT;
            set_layout_1.bindings.insert(0, bind);

            let mut set_layouts = renderer_set_0_layouts();
            set_layouts.push(set_layout_1);
            let infos = PipelineDescriptorSetLayoutCreateInfo {
                flags: Default::default(),
                set_layouts,
                push_constant_ranges: vec![PC_RANGE],
            };
            PipelineLayout::new(
                device.clone(),
                infos
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
                .unwrap()
        };

        let mut ci = GraphicsPipelineCreateInfo::layout(layout);
        ci.vertex_input_state = Some(vertex_input_state);
        ci.input_assembly_state = Some(InputAssemblyState::default());
        ci.viewport_state = Some(ViewportState::default());
        ci.rasterization_state = Some(RasterizationState::default());
        ci.color_blend_state = Some(ColorBlendState {
            attachments: vec![
                ColorBlendAttachmentState::default(),
                ColorBlendAttachmentState::default(),
                ColorBlendAttachmentState::default(),
            ],
            ..Default::default()
        });
        ci.multisample_state = Some(MultisampleState::default());
        ci.subpass = Some(subpass.into());
        ci.depth_stencil_state = Some(DepthStencilState {
            depth: Some(DepthState { write_enable: false, compare_op: CompareOp::Equal }),
            ..Default::default()
        });
        ci.stages = stages.into_iter().collect();
        ci.dynamic_state.insert(DynamicState::Viewport);
        ci.dynamic_state.insert(DynamicState::Scissor);

        GraphicsPipeline::new(device.clone(), None, ci).unwrap()
    };
    let mrt_instanced_pipeline = {
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 position;
                    layout(location = 1) in vec2 uvs;
                    layout(location = 2) in vec3 normal;

                    layout(location = 0) out vec3 v_normal;
                    layout(location = 1) out vec2 v_uvs;

                    layout(set=0, binding=0, std140) uniform UBO {
                        mat4 view;
                        mat4 projection;
                        mat4 inverse_projection;
                        vec4 sun_direction;
                    };

                    struct Transform {
                        mat4 t;
                    };
                    layout(set = 1, binding = 1, std140) readonly buffer SSBO {
                        Transform transforms[];
                    };

                    void main() {
                        int index = gl_InstanceIndex;

                        mat4 model = transforms[index].t;
                        mat3 normal_mat = transpose(inverse(mat3(model)));
                        vec3 world_norm = normalize(normal_mat * normal);

                        vec4 world_pos = model * vec4(position,1.0);
                        vec4 view_pos = view * world_pos;

                        v_normal = normalize((view * vec4(world_norm,0.0)).xyz);
                        v_uvs = uvs;

                        gl_Position = projection * view_pos;
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 v_normal;
                    layout(location = 1) in vec2 v_uvs;

                    layout(location = 0) out vec4 normal_mrt;
                    layout(location = 1) out vec4 uvs_mrt;
                    layout(location = 2) out vec4 albedo_mrt;

                    layout (set = 1, binding = 0) uniform sampler2D albedo_tex;

                    void main() {
                        normal_mrt = vec4(normalize(v_normal), 0.0);
                        uvs_mrt = vec4(v_uvs, 0.0, 0.0);
                        albedo_mrt = texture(albedo_tex, v_uvs);
                    }
                ",
            }
        }

        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![
                Some(Format::R16G16B16A16_SFLOAT),
                Some(Format::R16G16B16A16_SFLOAT),
                Some(Format::R16G16B16A16_SFLOAT),
            ],
            depth_attachment_format: Some(Format::D24_UNORM_S8_UINT),
            ..Default::default()
        };

        let layout = {
            let mut set_layout_1 = DescriptorSetLayoutCreateInfo::default();
            let mut bind = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
            bind.stages = ShaderStages::FRAGMENT;
            set_layout_1.bindings.insert(0, bind);

            let mut bind_ssbo = DescriptorSetLayoutBinding::descriptor_type(StorageBuffer);
            bind_ssbo.stages = ShaderStages::VERTEX;
            set_layout_1.bindings.insert(1, bind_ssbo);

            let mut set_layouts = renderer_set_0_layouts();
            set_layouts.push(set_layout_1);
            let infos = PipelineDescriptorSetLayoutCreateInfo {
                flags: Default::default(),
                set_layouts,
                push_constant_ranges: vec![],
            };
            PipelineLayout::new(
                device.clone(),
                infos
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
                .unwrap()
        };

        let mut ci = GraphicsPipelineCreateInfo::layout(layout);
        ci.vertex_input_state = Some(vertex_input_state);
        ci.input_assembly_state = Some(InputAssemblyState::default());
        ci.viewport_state = Some(ViewportState::default());
        ci.rasterization_state = Some(RasterizationState::default());
        ci.color_blend_state = Some(ColorBlendState {
            attachments: vec![
                ColorBlendAttachmentState::default(),
                ColorBlendAttachmentState::default(),
                ColorBlendAttachmentState::default(),
            ],
            ..Default::default()
        });
        ci.multisample_state = Some(MultisampleState::default());
        ci.subpass = Some(subpass.into());
        ci.depth_stencil_state = Some(DepthStencilState {
            depth: Some(DepthState {
                write_enable: false,
                compare_op: CompareOp::Equal, // ← because pre-depth already wrote it
            }),
            ..Default::default()
        });
        ci.stages = stages.into_iter().collect();
        ci.dynamic_state.insert(DynamicState::Viewport);
        ci.dynamic_state.insert(DynamicState::Scissor);

        GraphicsPipeline::new(device.clone(), None, ci).unwrap()
    };

    let window_size = swapchain_images[0].extent();

    // Create GBuffer Images
    let gbuffer_images: Vec<Arc<Image>> = (0..3)
        .map(|_| {
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::R16G16B16A16_SFLOAT,
                    extent: [window_size[0], window_size[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap()
        })
        .collect();

    let depth_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::D24_UNORM_S8_UINT,
            extent: [window_size[0], window_size[1], 1],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let to_view = |img: Arc<Image>| {
        ImageView::new(
            img.clone(),
            ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                format: img.format(),
                component_mapping: Default::default(),
                subresource_range: ImageSubresourceRange {
                    aspects: if img
                        .format_features()
                        .contains(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
                    {
                        ImageAspects::DEPTH
                    } else {
                        ImageAspects::COLOR
                    },
                    mip_levels: (0..img.mip_levels()),
                    array_layers: (0..img.array_layers()),
                },
                usage: Default::default(),
                sampler_ycbcr_conversion: None,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let mrt = MRT {
        predepth_pipeline,
        gbuffer_pipeline: mrt_pipeline,
        gbuffer_instanced_pipeline: mrt_instanced_pipeline,
        gbuffer_depth_view: to_view(depth_image.clone()),
        gbuffer_image_views: gbuffer_images
            .iter()
            .map(|img| to_view(img.clone()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    };

    // Create MRT Lighting output image
    let mrt_lighting_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_SFLOAT,
            extent: [window_size[0], window_size[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    // ==============================================================
    // === MRT LIGHTING PASS (UBO + GBUFFER INPUT)
    // ==============================================================
    let mrt_lighting = {
        let pass = {
            mod vs {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 450
                        layout(location = 0) out vec2 v_uvs;
                        void main() {
                            v_uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                            gl_Position = vec4(v_uvs * 2.0 + -1.0, 0.0, 1.0);
                        }
                    "
                }
            }

            mod fs {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 450
                        layout(location = 0) in vec2 v_uvs;

                        layout(set = 0, binding = 0) uniform UBO {
                            mat4 view;
                            mat4 projection;
                            mat4 inverse_projection;
                            vec4 sun_direction;
                        } ubo;

                        layout(set=1, binding=0) uniform sampler2D normal_tex;
                        layout(set=1, binding=1) uniform sampler2D uvs_tex;
                        layout(set=1, binding=2) uniform sampler2D albedo_tex;
                        layout(set=1, binding=3) uniform sampler2D depth_tex;

                        layout(location = 0) out vec4 out_hdr;

                        void main() {
                            float depth = texture(depth_tex, v_uvs).r;
                            if(depth >= 1.0) discard;

                            vec3 N = normalize(texture(normal_tex, v_uvs).xyz);
                            vec3 L = normalize(ubo.sun_direction.xyz);
                            float ndotl = max(dot(N,L), 0.0);

                            vec3 albedo = vec3(0.8, 0.9, 0.1) * texture(albedo_tex, v_uvs).xyz;

                            out_hdr = vec4(albedo * ndotl, 1.0);
                        }
                    "
                }
            }

            let vs = vs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let set_layout_ubo =
                DescriptorSetLayout::new(device.clone(), renderer_set_0_layouts()[0].clone())
                    .unwrap(); // UBO layout

            let set_layout_gb = DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings: BTreeMap::from([
                        (0, {
                            let mut bind =
                                DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
                            bind.stages = ShaderStages::FRAGMENT;

                            bind
                        }),
                        (1, {
                            let mut bind =
                                DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
                            bind.stages = ShaderStages::FRAGMENT;

                            bind
                        }),
                        (2, {
                            let mut bind =
                                DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
                            bind.stages = ShaderStages::FRAGMENT;

                            bind
                        }),
                        (3, {
                            let mut bind =
                                DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
                            bind.stages = ShaderStages::FRAGMENT;

                            bind
                        }),
                    ]),
                    ..Default::default()
                },
            )
            .unwrap();

            let gb_set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                set_layout_gb.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        mrt.gbuffer_image_views[0].clone(),
                        default_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        mrt.gbuffer_image_views[1].clone(),
                        default_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        2,
                        mrt.gbuffer_image_views[2].clone(),
                        default_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        3,
                        mrt.gbuffer_depth_view.clone(),
                        default_sampler.clone(),
                    ),
                ],
                [],
            )
            .unwrap();

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![set_layout_ubo.clone(), set_layout_gb.clone()],
                    push_constant_ranges: vec![],
                    ..Default::default()
                },
            )
            .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
                ..Default::default()
            };

            let mut ci = GraphicsPipelineCreateInfo::layout(layout.clone());
            ci.vertex_input_state = Some(VertexInputState::new());
            ci.input_assembly_state = Some(InputAssemblyState::default());
            ci.viewport_state = Some(ViewportState::default());
            ci.rasterization_state = Some(RasterizationState::default());
            ci.multisample_state = Some(MultisampleState::default());
            ci.color_blend_state = Some(ColorBlendState {
                attachments: [ColorBlendAttachmentState::default()].to_vec(),
                ..Default::default()
            });
            ci.subpass = Some(subpass.into());
            ci.stages.push(stages[0].clone());
            ci.stages.push(stages[1].clone());
            ci.dynamic_state.insert(DynamicState::Viewport);
            ci.dynamic_state.insert(DynamicState::Scissor);

            MRTLighting {
                pipeline: GraphicsPipeline::new(device.clone(), None, ci).unwrap(),
                set: gb_set,
                image_view: ImageView::new_default(mrt_lighting_image.clone()).unwrap(),
            }
        };
        pass
    };

    // ==============================================================
    // === Bloom PASS (From HDR → another HDR)
    // ==============================================================
    let bloom_pass =
        BloomPass::new(device, memory_allocator, window_size[0], window_size[1]).unwrap();

    // ==============================================================
    // === COMPOSITE PASS (From HDR → another HDR)
    // ==============================================================
    let composite_image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16G16B16A16_SFLOAT,
            extent: [window_size[0], window_size[1], 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let composite = {
        let pass = {
            mod vs {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 450
                        layout(location = 0) out vec2 v_uvs;
                        void main(){
                            v_uvs = vec2((gl_VertexIndex<<1)&2, gl_VertexIndex&2);
                            gl_Position = vec4(v_uvs*2.0 + -1.0,0.0,1.0);
                        }
                    "
                }
            }

            mod fs {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 450

                        layout(location = 0) in vec2 v_uvs;
                        layout(location = 0) out vec4 out_color;

                        layout(set=0, binding=0) uniform sampler2D hdr_tex;
                        layout(set=0, binding=1) uniform sampler2D bloom_tex;

                        layout (push_constant) uniform PC {
                            float bloom_strength;
                        };

                        void main() {
                            vec3 hdr = texture(hdr_tex, v_uvs).rgb;
                            vec3 bloom = texture(bloom_tex, v_uvs).rgb;

                            float bloom_final_strength = max(bloom_strength, 1.0);
                            out_color = vec4(hdr + bloom * bloom_final_strength, 1.0);
                        }
                    "
                }
            }

            let vs = vs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let set_layout = DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings: BTreeMap::from([
                        (0, {
                            let mut b =
                                DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
                            b.stages = ShaderStages::FRAGMENT;
                            b
                        }),
                        (1, {
                            let mut b =
                                DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
                            b.stages = ShaderStages::FRAGMENT;
                            b
                        }),
                    ]),
                    ..Default::default()
                },
            )
            .unwrap();

            let set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                set_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        ImageView::new_default(mrt_lighting_image.clone()).unwrap(),
                        default_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        bloom_pass.result(), // bloom mip 0 (after upsample accumulation)
                        default_sampler.clone(),
                    ),
                ],
                [],
            )
            .unwrap();
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![set_layout.clone()],
                    push_constant_ranges: [PushConstantRange {
                        stages: ShaderStages::FRAGMENT,
                        offset: 0,
                        size: std::mem::size_of::<fs::PC>() as _,
                    }]
                    .to_vec(),
                    ..Default::default()
                },
            )
            .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(Format::R16G16B16A16_SFLOAT)],
                ..Default::default()
            };

            let mut ci = GraphicsPipelineCreateInfo::layout(layout);
            ci.vertex_input_state = Some(VertexInputState::new());
            ci.input_assembly_state = Some(InputAssemblyState::default());
            ci.viewport_state = Some(ViewportState::default());
            ci.rasterization_state = Some(RasterizationState::default());
            ci.color_blend_state = Some(ColorBlendState {
                attachments: [ColorBlendAttachmentState::default()].to_vec(),
                ..Default::default()
            });
            ci.multisample_state = Some(MultisampleState::default());
            ci.subpass = Some(subpass.into());
            ci.stages.push(stages[0].clone());
            ci.stages.push(stages[1].clone());
            ci.dynamic_state.insert(DynamicState::Viewport);
            ci.dynamic_state.insert(DynamicState::Scissor);

            Composite {
                pipeline: GraphicsPipeline::new(device.clone(), None, ci).unwrap(),
                set,
                image_view: ImageView::new_default(composite_image.clone()).unwrap(),
            }
        };
        pass
    };

    // ==============================================================
    // === SWAPCHAIN PASS (unchanged)
    // ==============================================================
    let swapchain_pass = {
        let pass = {
            mod vs_fullscreen {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 450
                        layout(location = 0) out vec2 uvs;
                        void main() {
                            uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                            gl_Position = vec4(uvs * 2.0 - 1.0, 0.0, 1.0);
                        }
                    ",
                }
            }

            mod fs_fullscreen {
                vulkano_shaders::shader! {
                    ty: "fragment",
                    src: r"
                        #version 450
                        layout(location = 0) in vec2 uvs;
                        layout(location = 0) out vec4 f_color;

                        layout(set=0, binding=0) uniform sampler2D composite_output;

                        vec3 aces(vec3 x) {
                            return clamp((x*(2.51*x+0.03)) / (x*(2.43*x+0.59)+0.14), 0.0, 1.0);
                        }

                        vec3 linear_to_srgb(vec3 x){
                            return pow(x, vec3(1.0/2.2));
                        }

                        void main(){
                            vec3 color = texture(composite_output, uvs).xyz;
                            color = linear_to_srgb(aces(color));
                            f_color = vec4(color,1.0);
                        }
                    ",
                }
            }

            let vs = vs_fullscreen::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs_fullscreen::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let mut composite_image = DescriptorSetLayoutCreateInfo::default();
            composite_image.bindings.insert(0, {
                let mut bind = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
                bind.stages = ShaderStages::FRAGMENT;
                bind
            });

            let mut ci = GraphicsPipelineCreateInfo::layout({
                let info = PipelineDescriptorSetLayoutCreateInfo {
                    flags: Default::default(),
                    set_layouts: [composite_image].into(),
                    push_constant_ranges: vec![],
                };
                PipelineLayout::new(
                    device.clone(),
                    info.into_pipeline_layout_create_info(device.clone())
                        .unwrap(),
                )
                .unwrap()
            });

            ci.input_assembly_state = Some(InputAssemblyState::default());
            ci.vertex_input_state = Some(VertexInputState::new());
            ci.viewport_state = Some(ViewportState::default());
            ci.rasterization_state = Some(RasterizationState::default());
            ci.multisample_state = Some(MultisampleState::default());
            ci.color_blend_state = Some(ColorBlendState {
                attachments: [ColorBlendAttachmentState::default()].to_vec(),
                ..Default::default()
            });
            ci.subpass = Some(
                PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(Format::B8G8R8A8_SRGB)],
                    ..Default::default()
                }
                .into(),
            );
            ci.stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ]
            .into_iter()
            .collect();
            ci.dynamic_state.insert(DynamicState::Viewport);
            ci.dynamic_state.insert(DynamicState::Scissor);

            let pipeline = GraphicsPipeline::new(device.clone(), None, ci).unwrap();

            let mut bind = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
            bind.stages = ShaderStages::FRAGMENT;

            let descriptor_layout = DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    flags: Default::default(),
                    bindings: BTreeMap::from([(0, bind)]),
                    ..Default::default()
                },
            )
            .unwrap();

            let set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                descriptor_layout.clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    composite.image_view.clone(),
                    default_sampler.clone(),
                )],
                [],
            )
            .unwrap();

            SwapchainPass {
                pipeline,
                set,
                attachment_image_views: swapchain_images
                    .iter()
                    .map(|i| ImageView::new_default(i.clone()).unwrap())
                    .collect(),
            }
        };
        pass
    };

    let (cull_compute_pass, cull_prefix_pass) = construct_culling_passes(&device, &descriptor_set_allocator);

    (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass, cull_compute_pass, cull_prefix_pass)
}
