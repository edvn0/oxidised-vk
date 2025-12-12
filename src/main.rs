extern crate nalgebra_glm as glm;
mod bloom_pass;
mod camera;
mod input_state;
mod main_helpers;
mod math;
mod mesh;
mod shader_bindings;
mod texture_cache;
mod vertex;

use crate::bloom_pass::BloomPass;
use crate::camera::Camera;
use crate::input_state::InputState;
use crate::main_helpers::FrameDescriptorSet;
use crate::mesh::{MeshAsset, import_gltf, upload_build_mesh};
use crate::shader_bindings::{RendererUBO, renderer_set_0_layouts};
use crate::vertex::{PositionMeshVertex, StandardMeshVertex};
use nalgebra::{Matrix4, Translation3};
use rand::Rng;
use std::collections::{BTreeMap, HashMap};
use std::default::Default;
use std::time::Instant;
use std::{error::Error, sync::Arc};
use vulkano::command_buffer::{
    ClearColorImageInfo, DrawIndexedIndirectCommand, PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::layout::DescriptorType::{CombinedImageSampler, StorageBuffer};
use vulkano::device::DeviceOwned;
use vulkano::format::{ClearColorValue, ClearValue, FormatFeatures};
use vulkano::image::sampler::{
    BorderColor, Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode, SamplerReductionMode,
};
use vulkano::image::view::{ImageViewCreateInfo, ImageViewType};
use vulkano::image::{ImageAspects, ImageSubresourceRange, SampleCount};
use vulkano::instance::InstanceExtensions;
use vulkano::instance::debug::DebugUtilsLabel;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexInputState};
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
            vertex_input::VertexDefinition,
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
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const INSTANCE_COUNT: DeviceSize = 200;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop).unwrap();

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    _compute_queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    rcx: Option<RenderContext>,

    input_state: InputState,
    camera: Camera,
    last_frame: Instant,

    cull_backfaces: bool,
    clockwise_front_face: bool,
    lod_choice: usize,
}
struct MRT {
    predepth_pipeline: Arc<GraphicsPipeline>,
    gbuffer_image_views: [Arc<ImageView>; 3],
    gbuffer_depth_view: Arc<ImageView>,
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

struct FrameSubmission {
    transforms: Vec<TransformTRS>,
    material_ids: Vec<u32>,
    mesh_offsets: Vec<u32>,
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

    meshes: HashMap<String, Arc<MeshAsset>>,

    frame_transforms: Vec<Subbuffer<[TransformTRS]>>, // TODO: Move
    static_transforms: Vec<TransformTRS>,
    material_id_buffers: Vec<Subbuffer<[u32]>>,
    indirect_buffers: Vec<Subbuffer<[DrawIndexedIndirectCommand]>>,
    white_image_sampler: Arc<ImageView>,
    black_image_sampler: Arc<ImageView>,

    context_descriptor_set: FrameDescriptorSet,
    frame_uniform_buffers: Vec<Subbuffer<RendererUBO>>,

    frame_submission: FrameSubmission,

    mrt_pass: MRT,
    mrt_lighting: MRTLighting,
    composite: Composite,
    swapchain_pass: SwapchainPass,
    bloom_pass: BloomPass,
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

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_timeline_semaphore: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, graphics_index, compute_index) = {
            let mut best = None;
            for p in instance.enumerate_physical_devices().unwrap() {
                if !(p.api_version() >= Version::V1_4
                    || p.supported_extensions().khr_dynamic_rendering)
                {
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

                let g = match gq {
                    Some(v) => v,
                    None => continue,
                };
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
        infos.push(QueueCreateInfo {
            queue_family_index: graphics_index,
            ..Default::default()
        });
        if compute_index != graphics_index {
            infos.push(QueueCreateInfo {
                queue_family_index: compute_index,
                ..Default::default()
            });
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: infos,
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    timeline_semaphore: true,
                    shader_draw_parameters: true,
                    multi_draw_indirect: true,
                    buffer_device_address: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();

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

        Ok(App {
            instance,
            device,
            graphics_queue,
            _compute_queue: compute_queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            input_state: InputState::new(),
            camera: Camera::new(),
            last_frame: Instant::now(),
            rcx: None,
            cull_backfaces: true,
            clockwise_front_face: true,
            lod_choice: 0,
        })
    }
}

fn generate_random_transforms(count: u64) -> Vec<TransformTRS> {
    let mut rng = rand::rng();
    let mut transforms = Vec::with_capacity(count as usize);

    for _ in 0..count {
        let x = rng.random_range(-50.0..50.0);
        let y = rng.random_range(-50.0..50.0);
        let z = rng.random_range(-50.0..50.0);
        let translation = Translation3::new(x, y, z);

        let rx = rng.random_range(0.0..std::f32::consts::TAU);
        let ry = rng.random_range(0.0..std::f32::consts::TAU);
        let rz = rng.random_range(0.0..std::f32::consts::TAU);
        let rotation = nalgebra::Rotation3::from_euler_angles(rx, ry, rz);

        let scale = rng.random_range(1.0..5.0);
        let scale_matrix = Matrix4::new_scaling(scale);

        let transform = translation.to_homogeneous() * rotation.to_homogeneous() * scale_matrix;

        transforms.push(TransformTRS {
            trs: transform.as_slice().try_into().unwrap(),
        });
    }

    transforms
}

fn calculate_submission_requirements(
    meshes: &HashMap<String, Arc<MeshAsset>>,
    instance_count: usize,
) -> (usize, usize, usize) {
    let mut total_submeshes = 0;
    let mut total_submesh_lods = 0;

    for mesh in meshes.values() {
        total_submeshes += mesh.submeshes.len();
        total_submesh_lods += mesh.submeshes.len() * mesh.lods.len();
    }

    let transforms = instance_count;
    let material_ids = total_submeshes * instance_count;
    let mesh_offsets = total_submesh_lods + 1;

    (transforms, material_ids, mesh_offsets)
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut attribs = Window::default_attributes();
        attribs.inner_size = Some(PhysicalSize::new(1280, 1024).into());
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
                .map(|x| x.0)
                .find(|fmt| *fmt == Format::B8G8R8A8_SRGB || *fmt == Format::R8G8B8A8_SRGB)
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

        let white_tex = create_image(
            &self.graphics_queue,
            &self.command_buffer_allocator,
            &self.memory_allocator,
            0xFF,
        );
        let black_tex = create_image(
            &self.graphics_queue,
            &self.command_buffer_allocator,
            &self.memory_allocator,
            0x00,
        );

        let (built_asset, image_array) = import_gltf(
            "assets/viking_room.glb",
            &self.memory_allocator,
            &self.command_buffer_allocator,
            &self.graphics_queue,
            &white_tex,
            false,
        )
        .unwrap();

        let meshes = HashMap::from_iter([(
            "viking_room".to_string(),
            upload_build_mesh(built_asset, image_array, &self.memory_allocator).unwrap(),
        )]);
        let (t, m, o) = calculate_submission_requirements(&meshes, INSTANCE_COUNT as usize);

        let (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass) =
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
            depth_range: 0.0..=1.0,
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

        let frame_transforms = (0..image_count)
            .map(|_| {
                let max_total_instances = INSTANCE_COUNT as usize;

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
                    max_total_instances as DeviceSize,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let asset = &meshes["viking_room"];
        let lod = self.lod_choice;
        let draw_template = asset
            .submeshes
            .iter()
            .map(|sub| DrawIndexedIndirectCommand {
                index_count: sub.index_count,
                instance_count: 0,
                first_index: asset.lods[lod].first_index + sub.first_index,
                vertex_offset: asset.lods[lod].vertex_offset as u32,
                first_instance: 0,
            })
            .collect::<Vec<_>>();

        let indirect_buffers = (0..image_count)
            .map(|_| {
                Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDIRECT_BUFFER
                            | BufferUsage::STORAGE_BUFFER
                            | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    draw_template.clone(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let draw_count = meshes["viking_room"].lods.len() * meshes["viking_room"].submeshes.len();

        let material_id_buffers = (0..image_count)
            .map(|_| {
                Buffer::new_slice::<u32>(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    draw_count as DeviceSize,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

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

        let static_transforms = generate_random_transforms(INSTANCE_COUNT);

        for ssbo in frame_transforms.iter() {
            if let Ok(mut write) = ssbo.write() {
                let len = write.len().min(static_transforms.len());
                write[..len].copy_from_slice(&static_transforms[..len]);
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
            frame_transforms,
            static_transforms,
            material_id_buffers,
            indirect_buffers,
            white_image_sampler: white_tex,
            black_image_sampler: black_tex,
            frame_uniform_buffers: uniform_buffers,
            mrt_pass: mrt,
            mrt_lighting,
            composite,
            bloom_pass,

            frame_submission: FrameSubmission {
                transforms: Vec::with_capacity(t),
                material_ids: Vec::with_capacity(m),
                mesh_offsets: vec![0; o],
            },
            meshes,
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
                    && event.state == ElementState::Pressed
                    && !event.repeat
                {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyU) => {
                            self.cull_backfaces = !self.cull_backfaces;
                            println!("Culling: {:?}", self.cull_backfaces);
                        }
                        PhysicalKey::Code(KeyCode::KeyH) => {
                            self.clockwise_front_face = !self.clockwise_front_face;
                            println!("Winding: {:?}", self.clockwise_front_face);
                        }
                        PhysicalKey::Code(KeyCode::KeyY) => {
                            let lod_count =
                                self.rcx.as_ref().unwrap().meshes["viking_room"].lods.len();
                            self.lod_choice = (self.lod_choice + 1) % lod_count;
                            println!("LOD set to {}", self.lod_choice);
                        }
                        _ => {}
                    }
                }
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

    allocator
        .device()
        .set_debug_utils_object_name(&img, Some("White Renderer Texture"))
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

impl RenderContext {
    fn draw_mesh(
        &mut self,
        asset: &MeshAsset,
        lod: usize,
        transform: TransformTRS,
        override_material: Option<u32>,
    ) {
        debug_assert!(lod < asset.lods.len());
        debug_assert!(self.frame_submission.transforms.len() <= INSTANCE_COUNT as usize);

        let frame = &mut self.frame_submission;

        let object_index = frame.transforms.len() as u32;
        debug_assert!(object_index < INSTANCE_COUNT as u32);

        frame.transforms.push(transform);

        let submesh_count = asset.submeshes.len();
        let base_draw_index = lod * submesh_count;

        for (submesh_index, submesh) in asset.submeshes.iter().enumerate() {
            let mat = override_material.unwrap_or(submesh.material_index);
            frame.material_ids.push(mat);

            let draw_index = base_draw_index + submesh_index;
            frame.mesh_offsets[draw_index + 1] += 1;
        }
    }
}

impl App {
    fn render_frame(&mut self) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.frame_submission.transforms.clear();
        rcx.frame_submission.material_ids.clear();
        rcx.frame_submission.mesh_offsets.fill(0);

        let mesh = rcx.meshes.get("viking_room").unwrap().clone(); // or reference if Cheap
        let lod = self.lod_choice;
        for t in rcx.static_transforms.clone() {
            rcx.draw_mesh(&mesh, lod, t, None);
        }

        let offsets = &mut rcx.frame_submission.mesh_offsets;

        let mut acc = 0u32;
        for i in 0..offsets.len() {
            let v = offsets[i];
            offsets[i] = acc;
            acc += v;
        }

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

            let (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass) =
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

        acquire_future.wait(None).unwrap();

        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let mut graphics_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        rcx.update_camera_ubo(&self.camera, image_index as usize, window_size);

        let transforms = rcx.frame_transforms[image_index as usize].clone();
        let material_ids = rcx.material_id_buffers[image_index as usize].clone();

        {
            /* === TRANSFORMS === */
            if let Ok(mut w) = transforms.write() {
                let n = rcx.frame_submission.transforms.len().min(w.len());
                w[..n].copy_from_slice(&rcx.frame_submission.transforms[..n]);
            }

            if let Ok(mut w) = material_ids.write() {
                let mesh = &rcx.meshes["viking_room"];
                let lod_count = mesh.lods.len();

                let mut draw_id = 0;

                for _ in 0..lod_count {
                    for sub in &mesh.submeshes {
                        w[draw_id] = sub.material_index;
                        draw_id += 1;
                    }
                }
            }

            let indirect = rcx.indirect_buffers[image_index as usize].clone();
            let total_instances = rcx.frame_submission.transforms.len() as u32;

            if let Ok(mut cmds) = indirect.write() {
                for cmd in cmds.iter_mut() {
                    cmd.first_instance = 0;
                    cmd.instance_count = total_instances;
                }
            }
        }

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
            )
            .unwrap();

            let sets = [
                rcx.context_descriptor_set
                    .for_frame(image_index as usize)
                    .clone(),
                set,
            ];

            graphics_builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Predepth Z".to_string(),
                    color: [0.1, 0.1, 0.9, 1.0],
                    ..Default::default()
                })
                .unwrap()
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
                })
                .unwrap()
                .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                .unwrap()
                .set_scissor(0, [rcx.scissor].into_iter().collect())
                .unwrap()
                .set_depth_compare_op(CompareOp::GreaterOrEqual)
                .unwrap()
                .set_depth_write_enable(true)
                .unwrap()
                .set_depth_test_enable(true)
                .unwrap()
                .set_cull_mode(if self.cull_backfaces {
                    CullMode::Back
                } else {
                    CullMode::Front
                })
                .unwrap()
                .set_front_face(if self.clockwise_front_face {
                    FrontFace::Clockwise
                } else {
                    FrontFace::CounterClockwise
                })
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
                .bind_vertex_buffers(0, rcx.meshes["viking_room"].position_vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(rcx.meshes["viking_room"].index_buffer.clone())
                .unwrap();

            unsafe {
                graphics_builder
                    .draw_indexed_indirect(rcx.indirect_buffers[image_index as usize].clone())
                    .unwrap();
                graphics_builder
                    .end_rendering()
                    .unwrap()
                    .end_debug_utils_label()
                    .unwrap();
            }
        }

        {
            let layout = rcx
                .mrt_pass
                .gbuffer_instanced_pipeline
                .layout()
                .set_layouts()[1]
                .clone();

            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                layout,
                [
                    WriteDescriptorSet::image_view_sampler_array(
                        0,
                        0,
                        rcx.meshes["viking_room"]
                            .texture_array
                            .iter()
                            .cloned()
                            .map(|v| (v, rcx.default_sampler.clone()))
                            .chain(std::iter::repeat((
                                rcx.white_image_sampler.clone(),
                                rcx.default_sampler.clone(),
                            )))
                            .take(256),
                    ),
                    WriteDescriptorSet::buffer(
                        1,
                        rcx.frame_transforms[image_index as usize].clone(),
                    ),
                    WriteDescriptorSet::buffer(
                        2,
                        rcx.material_id_buffers[image_index as usize].clone(),
                    ),
                    WriteDescriptorSet::buffer(
                        3,
                        rcx.meshes["viking_room"].materials_buffer.clone(),
                    ),
                ],
                [],
            )
            .unwrap();

            let sets = [
                rcx.context_descriptor_set
                    .for_frame(image_index as usize)
                    .clone(),
                set,
            ];

            graphics_builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Instanced MRT Geometry".to_string(),
                    color: [0.99, 0.1, 0.1, 1.0],
                    ..Default::default()
                })
                .unwrap()
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
                .set_depth_compare_op(CompareOp::Equal)
                .unwrap()
                .set_depth_write_enable(true)
                .unwrap()
                .set_depth_test_enable(true)
                .unwrap()
                .set_cull_mode(if self.cull_backfaces {
                    CullMode::Back
                } else {
                    CullMode::Front
                })
                .unwrap()
                .set_front_face(if self.clockwise_front_face {
                    FrontFace::Clockwise
                } else {
                    FrontFace::CounterClockwise
                })
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
                .bind_vertex_buffers(0, rcx.meshes["viking_room"].vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(rcx.meshes["viking_room"].index_buffer.clone())
                .unwrap();

            unsafe {
                graphics_builder
                    .draw_indexed_indirect(rcx.indirect_buffers[image_index as usize].clone())
                    .unwrap();
                graphics_builder
                    .end_rendering()
                    .unwrap()
                    .end_debug_utils_label()
                    .unwrap();
            }
        }

        {
            // Pass 2: MRT Lighting
            let descriptor_sets = vec![
                rcx.context_descriptor_set
                    .for_frame(image_index as usize)
                    .clone(),
                rcx.mrt_lighting.set.clone(),
            ];

            graphics_builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "MRT Lighting".to_string(),
                    color: [0.1, 0.99, 0.9, 1.0],
                    ..Default::default()
                })
                .unwrap()
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

            unsafe {
                graphics_builder.draw(3, 1, 0, 0).unwrap();
                graphics_builder
                    .end_rendering()
                    .unwrap()
                    .end_debug_utils_label()
                    .unwrap();
            };
        }

        {
            // Pass 2.1: Bloom
            rcx.bloom_pass.run(
                &mut graphics_builder,
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

            graphics_builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Compositing".to_string(),
                    color: [0.99, 0.99, 0.0, 1.0],
                    ..Default::default()
                })
                .unwrap()
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

            unsafe {
                graphics_builder.draw(3, 1, 0, 0).unwrap();
                graphics_builder
                    .end_rendering()
                    .unwrap()
                    .end_debug_utils_label()
                    .unwrap();
            };
        }

        {
            // Pass 4: Present, HDR -> LDR with tonemapping

            let descriptor_sets = vec![rcx.swapchain_pass.set.clone()];

            graphics_builder
                .begin_debug_utils_label(DebugUtilsLabel {
                    label_name: "Presentation".to_string(),
                    color: [0.99, 0.0, 0.75, 1.0],
                    ..Default::default()
                })
                .unwrap()
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

            unsafe {
                graphics_builder.draw(3, 1, 0, 0).unwrap();

                graphics_builder
                    .end_rendering()
                    .unwrap()
                    .end_debug_utils_label()
                    .unwrap();
            };
        }

        let cmd_buf = graphics_builder.build().unwrap();

        let future = rcx
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), cmd_buf)
            .unwrap()
            .then_swapchain_present(
                self.graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(f) => rcx.previous_frame_end = Some(f.boxed()),
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                eprintln!("present failed: {e}");
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }

        self.input_state.end_frame();
    }
}

fn window_size_dependent_setup(
    device: &Arc<Device>,
    memory_allocator: &Arc<GenericMemoryAllocator<FreeListAllocator>>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    default_sampler: &Arc<Sampler>,
    swapchain_images: &[Arc<Image>],
) -> (SwapchainPass, MRT, MRTLighting, Composite, BloomPass) {
    let predepth_pipeline = {
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                vulkan_version: "1.3",
                spirv_version: "1.6",
                path: "assets/shaders/predepth.vert",
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

        let vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = PositionMeshVertex::per_vertex().definition(&vs).unwrap();

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
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
            )
            .unwrap()
        };

        let mut ci = GraphicsPipelineCreateInfo::layout(layout);
        ci.vertex_input_state = Some(vertex_input_state);
        ci.input_assembly_state = Some(InputAssemblyState::default());
        ci.viewport_state = Some(ViewportState::default());
        ci.rasterization_state = Some(RasterizationState {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: Default::default(),
            cull_mode: CullMode::Back,
            front_face: FrontFace::Clockwise,
            depth_bias: None,
            line_width: 1.0,
            line_rasterization_mode: Default::default(),
            line_stipple: None,
            conservative: None,
            ..Default::default()
        });
        ci.multisample_state = Some(MultisampleState::default());
        ci.depth_stencil_state = Some(DepthStencilState {
            depth: Some(DepthState::reverse()), // Important
            ..Default::default()
        });
        ci.subpass = Some(subpass.into());
        ci.stages = stages.into_iter().collect();
        ci.dynamic_state.insert(DynamicState::Viewport);
        ci.dynamic_state.insert(DynamicState::Scissor);
        ci.dynamic_state.insert(DynamicState::DepthTestEnable);
        ci.dynamic_state.insert(DynamicState::DepthCompareOp);
        ci.dynamic_state.insert(DynamicState::DepthWriteEnable);
        ci.dynamic_state.insert(DynamicState::CullMode);
        ci.dynamic_state.insert(DynamicState::FrontFace);

        GraphicsPipeline::new(device.clone(), None, ci).unwrap()
    };

    let mrt_instanced_pipeline = {
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                vulkan_version: "1.3",
                spirv_version: "1.6",
                path: "assets/shaders/instanced_mrt.vert",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                vulkan_version: "1.3",
                spirv_version: "1.6",
                path: "./assets/shaders/instanced_mrt.frag",
                include: ["./assets/shaders/include"],
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

        let vertex_input_state = StandardMeshVertex::per_vertex().definition(&vs).unwrap();

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

            /* binding 0: sampler2D textures[] */
            let mut tex_bind = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
            tex_bind.stages = ShaderStages::FRAGMENT;
            tex_bind.descriptor_count = 2.max(256); // TODO! FIX!
            set_layout_1.bindings.insert(0, tex_bind);

            /* binding 1: Transform SSBO */
            let mut transform_bind = DescriptorSetLayoutBinding::descriptor_type(StorageBuffer);
            transform_bind.stages = ShaderStages::VERTEX;
            set_layout_1.bindings.insert(1, transform_bind);

            /* binding 2: material_ids[] */
            let mut matid_bind = DescriptorSetLayoutBinding::descriptor_type(StorageBuffer);
            matid_bind.stages = ShaderStages::VERTEX;
            set_layout_1.bindings.insert(2, matid_bind);

            /* binding 3: GpuMaterial[] */
            let mut mat_bind = DescriptorSetLayoutBinding::descriptor_type(StorageBuffer);
            mat_bind.stages = ShaderStages::FRAGMENT;
            set_layout_1.bindings.insert(3, mat_bind);

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
        ci.rasterization_state = Some(RasterizationState {
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            polygon_mode: Default::default(),
            cull_mode: CullMode::Back,
            front_face: FrontFace::Clockwise,
            depth_bias: None,
            line_width: 1.0,
            line_rasterization_mode: Default::default(),
            line_stipple: None,
            conservative: None,
            ..Default::default()
        });
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
        ci.dynamic_state.insert(DynamicState::DepthTestEnable);
        ci.dynamic_state.insert(DynamicState::DepthCompareOp);
        ci.dynamic_state.insert(DynamicState::DepthWriteEnable);
        ci.dynamic_state.insert(DynamicState::CullMode);
        ci.dynamic_state.insert(DynamicState::FrontFace);

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
                    path: "assets/shaders/mrt_lighting.frag",
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
            ci.dynamic_state.insert(DynamicState::DepthTestEnable);
            ci.dynamic_state.insert(DynamicState::DepthCompareOp);
            ci.dynamic_state.insert(DynamicState::DepthWriteEnable);

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
            ci.dynamic_state.insert(DynamicState::DepthTestEnable);
            ci.dynamic_state.insert(DynamicState::DepthCompareOp);
            ci.dynamic_state.insert(DynamicState::DepthWriteEnable);

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
            ci.dynamic_state.insert(DynamicState::DepthTestEnable);
            ci.dynamic_state.insert(DynamicState::DepthCompareOp);
            ci.dynamic_state.insert(DynamicState::DepthWriteEnable);

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

    // let (cull_compute_pass, cull_prefix_pass) = construct_culling_passes(&device);

    (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass)
}
