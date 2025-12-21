extern crate nalgebra_glm as glm;
mod bloom_pass;
mod camera;
mod components;
mod engine_shaders;
mod image;
mod imgui;
mod input_state;
mod main_helpers;
mod math;
mod mesh;
mod mesh_registry;
mod render_context;
mod render_passes;
mod scene;
mod shader_bindings;
mod submission;
mod texture_cache;
mod vertex;
mod windowing;

use crate::bloom_pass::{BloomPass, BloomSettings};
use crate::camera::Camera;
use crate::components::{MeshComponent, Visible};
use crate::image::{ImageDimensions, ImageInfo, create_image};
use crate::imgui::renderer::ImGuiRenderer;
use crate::input_state::InputState;
use crate::main_helpers::{FrameDescriptorSet, generate_identity_lut_3d};
use crate::mesh::{ImageViewSampler, MeshAsset, load_meshes_from_directory};
use crate::render_context::{
    Culling, FrameResources, MeshDrawStream, RenderContext, TransformTRS, Winding,
};
use crate::render_passes::data::{FrameContext, RenderResources};
use crate::render_passes::passes::{
    BloomEffectPass, CompositePass, GBufferPass, ImGuiPass, MRTLightingPass, PreDepthPass,
    PresentPass, RenderPass,
};
use crate::render_passes::recorder::RenderRecorder;
use crate::render_passes::recordings::{
    Composite, CompositeSettings, MRT, MRTLighting, SwapchainPass,
};
use crate::scene::Scene;
use crate::shader_bindings::{RendererUBO, renderer_set_0_layouts};
use crate::submission::FrameSubmission;
use crate::vertex::{PositionMeshVertex, StandardMeshVertex};
use crate::windowing::set_window_icons;
use ::imgui::{Condition, Context};
use glm::vec3;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use nalgebra::{Translation3, UnitQuaternion};
use rand::Rng;
use std::collections::{BTreeMap, HashMap};
use std::default::Default;
use std::path::Path;
use std::time::Instant;
use std::{error::Error, sync::Arc};
use vulkano::command_buffer::DrawIndexedIndirectCommand;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocatorCreateInfo;
use vulkano::descriptor_set::layout::DescriptorType::{CombinedImageSampler, StorageBuffer};
use vulkano::format::FormatFeatures;
use vulkano::image::sampler::{
    BorderColor, Filter, LOD_CLAMP_NONE, Sampler, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode,
};
use vulkano::image::view::{ImageViewCreateInfo, ImageViewType};
use vulkano::image::{ImageAspects, ImageSubresourceRange};
use vulkano::instance::InstanceExtensions;
use vulkano::pipeline::graphics::depth_stencil::{CompareOp, DepthState, DepthStencilState};
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexInputState};
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::{
    DeviceSize, Validated, Version, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, allocator::StandardCommandBufferAllocator,
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
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
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

const INSTANCE_COUNT: DeviceSize = 500;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop).unwrap();

    event_loop.run_app(&mut app)
}

#[derive(Clone)]
pub struct GpuUploadContext {
    pub queue: Arc<Queue>,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub memory_allocator: Arc<StandardMemoryAllocator>,
}

enum PostProcessPanel {
    Bloom,
    Composite,
}

struct AppUIState {
    post_process_panel: PostProcessPanel,
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    gpu_upload: GpuUploadContext,
    _compute_queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    rcx: Option<RenderContext>,
    ui_state: AppUIState,

    input_state: InputState,
    camera: Camera,
    last_frame: Instant,
    scene: Scene,

    cull_backfaces: Culling,
    clockwise_front_face: Winding,
    lod_choice: usize,
}

#[repr(C)]
#[derive(BufferContents, Clone, Copy)]
pub struct FrustumPlanes {
    pub planes: [[f32; 4]; 6],
}

const MAX_FRAMES_IN_FLIGHT: usize = 3;

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
                    sampler_anisotropy: true,
                    dynamic_rendering: true,
                    timeline_semaphore: true,
                    shader_draw_parameters: true,
                    multi_draw_indirect: true,
                    buffer_device_address: true,
                    runtime_descriptor_array: true,
                    descriptor_binding_partially_bound: true,
                    shader_sampled_image_array_non_uniform_indexing: true,
                    shader_sampled_image_array_dynamic_indexing: true,
                    descriptor_binding_sampled_image_update_after_bind: true,
                    shader_int64: true,
                    shader_int16: true,
                    shader_int8: true,
                    shader_float16: true,
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
            StandardDescriptorSetAllocatorCreateInfo {
                update_after_bind: true,
                ..Default::default()
            },
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let gpu_upload = GpuUploadContext {
            queue: graphics_queue.clone(),
            command_buffer_allocator: command_buffer_allocator.clone(),
            memory_allocator: memory_allocator.clone(),
        };

        Ok(App {
            instance,
            device,
            graphics_queue,
            gpu_upload,
            _compute_queue: compute_queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            input_state: InputState::new(),
            camera: Camera::new(),
            last_frame: Instant::now(),
            rcx: None,
            cull_backfaces: Culling::Back,
            clockwise_front_face: Winding::CounterClockwise,
            lod_choice: 0,
            scene: Scene::new(),
            ui_state: AppUIState {
                post_process_panel: PostProcessPanel::Bloom,
            },
        })
    }
}

fn generate_random_transforms(count: u64) -> Vec<components::Transform> {
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
        let rotation = UnitQuaternion::from_euler_angles(rx, ry, rz).into_inner();

        let scale = rng.random_range(1.0..5.0);

        transforms.push(components::Transform {
            position: translation.vector,
            rotation,
            scale: vec3(scale, scale, scale),
        });
    }

    transforms
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_visible(false)
                        .with_inner_size(PhysicalSize::new(1600, 800)),
                )
                .unwrap(),
        );
        window.set_title("Oxidised");
        set_window_icons(&window);

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
                    min_image_count: surface_capabilities
                        .min_image_count
                        .max(MAX_FRAMES_IN_FLIGHT as u32),

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

        const MIP_COUNT: f32 = 16.0;
        let default_sampler = Sampler::new(
            self.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Linear,
                address_mode: [
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::Repeat,
                    SamplerAddressMode::Repeat,
                ],
                anisotropy: Some(MIP_COUNT),
                lod: 0.0..=LOD_CLAMP_NONE,
                ..Default::default()
            },
        )
        .unwrap();

        let white_as_u8_slice = [0xFF, 0xFF, 0xFF, 0xFF];
        let image_info = ImageInfo::white_texture(default_sampler.clone());
        let white_tex = create_image(
            self.graphics_queue.clone(),
            self.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            &white_as_u8_slice,
            image_info,
        );
        let black_as_u8_slice = [0x00, 0x00, 0x00, 0x00];
        let black_image_info = ImageInfo::black_texture(default_sampler.clone());
        let black_tex = create_image(
            self.graphics_queue.clone(),
            self.command_buffer_allocator.clone(),
            self.memory_allocator.clone(),
            &black_as_u8_slice,
            black_image_info,
        );

        let mesh_registry = load_meshes_from_directory(
            "assets/meshes",
            &self.memory_allocator,
            &self.command_buffer_allocator,
            &self.graphics_queue,
            white_tex.clone(),
        );

        let mut mesh_streams = HashMap::new();

        let meshes = mesh_registry.clone();
        for (handle, mesh) in meshes.read().unwrap().iter() {
            let draw_count = mesh.lods.len() * mesh.submeshes.len();

            let indirect: [Subbuffer<_>; MAX_FRAMES_IN_FLIGHT] = std::array::from_fn(|_| {
                Buffer::from_iter(
                    self.memory_allocator.clone(),
                    BufferCreateInfo {
                        usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::STORAGE_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                            | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                        ..Default::default()
                    },
                    mesh.submeshes.iter().map(|sub| DrawIndexedIndirectCommand {
                        index_count: sub.index_count,
                        instance_count: 0,
                        first_index: sub.first_index,
                        vertex_offset: 0,
                        first_instance: 0,
                    }),
                )
                .unwrap()
            });

            let material_ids = Buffer::new_slice::<u32>(
                self.memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                draw_count as DeviceSize,
            )
            .unwrap();

            {
                let mut w = material_ids.write().unwrap();
                update_material_ids_for_mesh(mesh, &mut w);
            }

            let transforms: [Subbuffer<[TransformTRS]>; MAX_FRAMES_IN_FLIGHT] =
                std::array::from_fn(|_| {
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
                        1,
                    )
                    .unwrap()
                });

            mesh_streams.insert(
                handle,
                MeshDrawStream::new(mesh.clone(), material_ids, indirect, transforms),
            );
        }

        let (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass) =
            window_size_dependent_setup(
                &self.device,
                &self.gpu_upload,
                &self.descriptor_set_allocator,
                white_tex.clone(),
                black_tex.clone(),
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

        let uniform_buffers = (0..MAX_FRAMES_IN_FLIGHT)
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

        {
            let registry = mesh_registry.read().unwrap();

            let viking_handle = registry.resolve("viking_room").unwrap();
            let suzanne_handle = registry.resolve("suzanne").unwrap();

            for trs in generate_random_transforms(INSTANCE_COUNT / 2) {
                self.scene.add_entity((
                    trs,
                    MeshComponent {
                        mesh: viking_handle,
                    },
                    Visible,
                ));
            }

            for trs in generate_random_transforms(INSTANCE_COUNT / 2) {
                self.scene.add_entity((
                    trs,
                    MeshComponent {
                        mesh: suzanne_handle,
                    },
                    Visible,
                ));
            }

            self.scene
                .save_to_file(Path::new("scene_save.scene"))
                .unwrap();
            self.scene = Scene::load_from_file(Path::new("scene_save.scene")).unwrap();
        }
        let mut imgui = Context::create();
        imgui.set_ini_filename(None);

        let mut renderer = ImGuiRenderer::new(
            self.device.clone(),
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            swapchain.image_format(),
        );

        renderer.upload_font_atlas(
            &mut imgui,
            self.graphics_queue.clone(),
            self.command_buffer_allocator.clone(),
        );

        let mut platform = WinitPlatform::new(&mut imgui); // step 1
        platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default); // step 2

        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| FrameResources::new(&uniform_buffers[i]))
            .collect();

        window.set_visible(true);

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
                &uniform_buffers[..MAX_FRAMES_IN_FLIGHT],
            ),
            white_image_sampler: white_tex,
            black_image_sampler: black_tex,
            mrt_pass: mrt,
            mrt_lighting,
            composite,
            composite_settings: CompositeSettings::default(),
            bloom_pass,
            bloom_settings: BloomSettings::default(),

            frame_submission: FrameSubmission { draws: vec![] },
            meshes: mesh_registry,
            mesh_streams,

            winit_platform: platform,
            imgui_context: imgui,
            imgui_renderer: renderer,
            current_frame: 0,
            frames,
        });
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        unsafe {
            self.device.wait_idle().unwrap();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        rcx.winit_platform.handle_event(
            rcx.imgui_context.io_mut(),
            rcx.window.as_ref(),
            &winit::event::Event::WindowEvent::<()> {
                window_id: rcx.window.id(),
                event: event.clone(),
            },
        );

        match &event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::KeyboardInput { .. } => {
                if let WindowEvent::KeyboardInput { event, .. } = &event
                    && event.state == ElementState::Pressed
                    && !event.repeat
                {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyU) => {
                            self.cull_backfaces = self.cull_backfaces.next();
                            println!("Culling: {:?}", self.cull_backfaces);
                        }
                        PhysicalKey::Code(KeyCode::KeyH) => {
                            self.clockwise_front_face = self.clockwise_front_face.toggle();
                            println!("Winding: {:?}", self.clockwise_front_face);
                        }
                        PhysicalKey::Code(KeyCode::KeyY) => {
                            let registry = self.rcx.as_ref().unwrap().meshes.read().unwrap();

                            let lod_count = registry
                                .resolve("viking_room")
                                .map(|m| registry.get(m))
                                .unwrap()
                                .lods
                                .len();

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
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && self.input_state.rotating
        {
            self.input_state.mouse_delta = (dx as f32, dy as f32);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();

        let now = Instant::now();
        let delta = now - self.last_frame;
        self.last_frame = now;

        rcx.imgui_context.io_mut().update_delta_time(delta);

        rcx.winit_platform
            .prepare_frame(rcx.imgui_context.io_mut(), rcx.window.as_ref())
            .expect("imgui prepare_frame");

        rcx.window.request_redraw();
    }
}

impl App {
    fn render_frame(&mut self) {
        let rcx = self.rcx.as_mut().unwrap();

        let ui = rcx.imgui_context.new_frame();
        ui.window("Hello world")
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(|| {
                ui.text("Hello world!");
                ui.text("こんにちは世界！");
                ui.text("This...is...imgui-rs!");
                ui.text(format!("FPS: {:?}", ui.io().framerate));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
            });

        ui.window("Post Processing").build(|| {
            let preview = match self.ui_state.post_process_panel {
                PostProcessPanel::Bloom => "Bloom",
                PostProcessPanel::Composite => "Composite",
            };

            if let Some(_cb) = ui.begin_combo("Effect", preview) {
                if ui.selectable("Bloom") {
                    self.ui_state.post_process_panel = PostProcessPanel::Bloom;
                }
                if ui.selectable("Composite") {
                    self.ui_state.post_process_panel = PostProcessPanel::Composite;
                }
            }

            ui.separator();

            match self.ui_state.post_process_panel {
                PostProcessPanel::Bloom => {
                    rcx.bloom_settings.ui(ui);
                }
                PostProcessPanel::Composite => {
                    rcx.composite_settings.ui(ui);
                }
            }
        });
        rcx.winit_platform.prepare_render(ui, &rcx.window);
        rcx.frame_submission.clear_all();

        self.scene.update();

        let mut submission = self
            .scene
            .resources_mut()
            .get_mut::<FrameSubmission>()
            .unwrap();

        submission.drain_draws_into(&mut rcx.frame_submission.draws);
        rcx.build_frame(self.memory_allocator.clone());

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        self.camera.update_from_input(&self.input_state, dt);

        let window_size = rcx.window.inner_size();
        rcx.elapsed_millis = rcx.start_time.elapsed().as_millis() as u64;

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

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
                    &self.gpu_upload,
                    &self.descriptor_set_allocator,
                    rcx.white_image_sampler.clone(),
                    rcx.black_image_sampler.clone(),
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

        let mut graphics_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.graphics_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        rcx.update_camera_ubo(&self.camera, rcx.current_frame, window_size);

        let frame = FrameContext {
            _frame_index: rcx.current_frame,
            viewport: &rcx.viewport,
            scissor: &rcx.scissor,
            culling: self.cull_backfaces,
            winding: self.clockwise_front_face,
        };

        let mut recorder = RenderRecorder {
            cmd: &mut graphics_builder,
            imgui_context: &mut rcx.imgui_context,
            imgui_renderer: &mut rcx.imgui_renderer,
        };
        let frame_descriptor_set = rcx.context_descriptor_set.for_frame(rcx.current_frame);

        let resources = RenderResources {
            mesh_streams: &rcx.mesh_streams,
            descriptor_sets: &self.descriptor_set_allocator,
            white_sampler: &rcx.white_image_sampler,
            swapchain_views: &rcx.swapchain_pass.attachment_image_views,
            viewport: &rcx.viewport,
            scissor: &rcx.scissor,
            current_frame: rcx.current_frame,
            frame_descriptor_set: frame_descriptor_set.clone(),
        };

        let passes: [&dyn RenderPass; _] = [
            &PreDepthPass { mrt: &rcx.mrt_pass },
            &GBufferPass { mrt: &rcx.mrt_pass },
            &MRTLightingPass {
                lighting: &rcx.mrt_lighting,
            },
            &BloomEffectPass {
                bloom: &rcx.bloom_pass,
                settings: &rcx.bloom_settings,
                input_image: &rcx.mrt_lighting.image_view,
            },
            &CompositePass {
                composite: &rcx.composite,
                bloom_enabled: rcx.bloom_settings.enabled,
                settings: &rcx.composite_settings,
            },
            &PresentPass {
                swapchain: &rcx.swapchain_pass,
                image_index: image_index.try_into().unwrap(),
            },
            &ImGuiPass {
                image_index: image_index.try_into().unwrap(),
            },
        ];

        for pass in passes {
            pass.record(&mut recorder, &frame, &resources).unwrap();
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

        rcx.current_frame = (rcx.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}

fn window_size_dependent_setup(
    device: &Arc<Device>,
    upload: &GpuUploadContext,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
    default_white_texture: Arc<ImageViewSampler>,
    default_black_texture: Arc<ImageViewSampler>,
    swapchain_images: &[Arc<Image>],
) -> (SwapchainPass, MRT, MRTLighting, Composite, BloomPass) {
    let predepth_pipeline = {
        let vs = engine_shaders::predepth::vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = engine_shaders::predepth::fs::load(device.clone())
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
        let vs = engine_shaders::mrt::vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = engine_shaders::mrt::fs::load(device.clone())
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
                upload.memory_allocator.clone(),
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
        upload.memory_allocator.clone(),
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

    let mrt = MRT::new(
        predepth_pipeline,
        mrt_instanced_pipeline,
        to_view(depth_image.clone()),
        gbuffer_images
            .iter()
            .map(|img| to_view(img.clone()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
    );

    // Create MRT Lighting output image
    let mrt_lighting_image = Image::new(
        upload.memory_allocator.clone(),
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
        let vs = engine_shaders::fullscreen_vertex_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = engine_shaders::mrt_light::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let set_layout_ubo =
            DescriptorSetLayout::new(device.clone(), renderer_set_0_layouts()[0].clone()).unwrap(); // UBO layout

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
                    default_white_texture.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    mrt.gbuffer_image_views[1].clone(),
                    default_white_texture.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    2,
                    mrt.gbuffer_image_views[2].clone(),
                    default_white_texture.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    3,
                    mrt.gbuffer_depth_view.clone(),
                    default_white_texture.sampler.clone(),
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

        MRTLighting::new(
            GraphicsPipeline::new(device.clone(), None, ci).unwrap(),
            gb_set,
            ImageView::new_default(mrt_lighting_image.clone()).unwrap(),
        )
    };

    // ==============================================================
    // === Bloom PASS (From HDR → another HDR)
    // ==============================================================
    let bloom_pass = BloomPass::new(
        device.clone(),
        upload.memory_allocator.clone(),
        window_size[0],
        window_size[1],
    )
    .unwrap();

    // ==============================================================
    // === COMPOSITE PASS (From HDR → another HDR)
    // ==============================================================
    let composite_image = Image::new(
        upload.memory_allocator.clone(),
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
        let vs = engine_shaders::fullscreen_vertex_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = engine_shaders::composite::load(device.clone())
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
                    default_white_texture.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    bloom_pass.result(), // bloom mip 0 (after upsample accumulation)
                    default_white_texture.sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        let disabled_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            set_layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    ImageView::new_default(mrt_lighting_image.clone()).unwrap(),
                    default_white_texture.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    default_black_texture.view.clone(),
                    default_black_texture.sampler.clone(),
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
                    size: std::mem::size_of::<engine_shaders::composite::PC>() as _,
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
        Composite::new(
            GraphicsPipeline::new(device.clone(), None, ci).unwrap(),
            set,
            disabled_set,
            ImageView::new_default(composite_image.clone()).unwrap(),
        )
    };

    // ==============================================================
    // === SWAPCHAIN PASS (unchanged)
    // ==============================================================
    let swapchain_pass = {
        let vs = engine_shaders::fullscreen_vertex_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = engine_shaders::fullscreen::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let mut composite_image = DescriptorSetLayoutCreateInfo::default();
        composite_image.bindings.insert(0, {
            let mut bind = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
            bind.stages = ShaderStages::FRAGMENT;
            bind
        });
        composite_image.bindings.insert(1, {
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

        let mut bind_0 = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
        bind_0.stages = ShaderStages::FRAGMENT;

        let mut bind_1 = DescriptorSetLayoutBinding::descriptor_type(CombinedImageSampler);
        bind_1.stages = ShaderStages::FRAGMENT;

        let descriptor_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                flags: Default::default(),
                bindings: BTreeMap::from([(0, bind_0), (1, bind_1)]),
                ..Default::default()
            },
        )
        .unwrap();

        let lut_size: u32 = 32;

        let value = generate_identity_lut_3d(lut_size);

        let lut_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                unnormalized_coordinates: false,
                ..Default::default()
            },
        )
        .unwrap();

        let image_info = ImageInfo {
            dimensions: ImageDimensions::Dim3d([lut_size, lut_size, lut_size]),
            format: Format::R8G8B8A8_UNORM,
            mips: Some(1),
            sampler: lut_sampler.clone(),
            debug_name: "identity_lut_3d".to_string(),
        };

        let lut_texture = create_image(
            upload.queue.clone(),
            upload.command_buffer_allocator.clone(),
            upload.memory_allocator.clone(),
            &value,
            image_info,
        );

        let set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            descriptor_layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(
                    0,
                    composite.image_view.clone(),
                    default_white_texture.sampler.clone(),
                ),
                WriteDescriptorSet::image_view_sampler(
                    1,
                    lut_texture.view.clone(),
                    lut_texture.sampler.clone(),
                ),
            ],
            [],
        )
        .unwrap();

        SwapchainPass::new(
            pipeline,
            set,
            swapchain_images
                .iter()
                .map(|i| ImageView::new_default(i.clone()).unwrap())
                .collect(),
        )
    };

    // let (cull_compute_pass, cull_prefix_pass) = construct_culling_passes(&device);

    (swapchain_pass, mrt, mrt_lighting, composite, bloom_pass)
}

fn update_material_ids_for_mesh(mesh: &MeshAsset, material_ids: &mut [u32]) {
    let mut draw_id = 0;

    for _lod in 0..mesh.lods.len() {
        for sub in &mesh.submeshes {
            material_ids[draw_id] = sub.material_index;
            draw_id += 1;
        }
    }
}
