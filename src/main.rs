use nalgebra::{Isometry3, Matrix4, Perspective3, Point3, Vector3};
use std::collections::BTreeMap;
use std::default::Default;
use std::{error::Error, sync::Arc};
use std::ops::Neg;
use glm::{TVec, TVec3};
use vulkano::image::sampler::{
    BorderColor, Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
    SamplerMipmapMode, SamplerReductionMode,
};
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::{
    buffer::{
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer
        ,
    }, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo,
        RenderingInfo,
    }, descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding,
            DescriptorSetLayoutCreateInfo, DescriptorType,
        },
        DescriptorSet,
        WriteDescriptorSet,
    }, device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage},
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
            DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
        }, Instance,
        InstanceCreateInfo,
    },
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        }, layout::PipelineDescriptorSetLayoutCreateInfo, DynamicState, GraphicsPipeline, Pipeline,
        PipelineBindPoint,
        PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    shader::ShaderStages,
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated,
    Version,
    VulkanError,
    VulkanLibrary,
};
use vulkano::format::ClearValue;
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::layout::{PipelineLayoutCreateFlags, PushConstantRange};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
extern crate nalgebra_glm as glm;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

fn proj_matrix(aspect: f32, fov_y_rad: f32, z_near: f32) -> Matrix4<f32> {
    let p = Perspective3::new(aspect, fov_y_rad, z_near, f32::INFINITY).to_homogeneous();
    Matrix4::new(
        p[(0,0)], 0.0,    0.0,   0.0,
        0.0,      p[(1,1)],0.0,   0.0,
        0.0,      0.0,    0.0,   -1.0,
        0.0,      0.0,    p[(2,3)], p[(3,3)]
    )
}

struct FrameDescriptorSet {
    sets: Vec<Arc<DescriptorSet>>,
}

impl FrameDescriptorSet {
    fn new(
        device: Arc<Device>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        uniform_buffers: &[Subbuffer<RendererUBO>],
    ) -> Self {
        let mut layout_binding =
            DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);
        layout_binding.stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
        let mut create_info = DescriptorSetLayoutCreateInfo::default();

        create_info.bindings.insert(0, layout_binding);
        let layout = DescriptorSetLayout::new(device.clone(), create_info).unwrap();

        let sets: Vec<_> = uniform_buffers
            .iter()
            .map(|ub| {
                let mut writes = Vec::new();
                writes.push(WriteDescriptorSet::buffer(0, ub.clone()));
                DescriptorSet::new(descriptor_set_allocator.clone(), layout.clone(), writes, [])
                    .unwrap()
            })
            .collect();

        Self { sets }
    }

    fn for_frame(&self, index: usize) -> Arc<DescriptorSet> {
        self.sets[index].clone()
    }
}

#[repr(C)]
#[derive(BufferContents)]
struct PushConsts {
    model: [f32; 16],
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    index_buffer: Subbuffer<[u32]>,
    rcx: Option<RenderContext>,
}

struct MRT {
    gbuffer_image_views: Vec<Arc<ImageView>>,
    gbuffer_pipeline: Arc<GraphicsPipeline>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,
    swapchain_pipeline: Arc<GraphicsPipeline>,
    swapchain_set: Arc<DescriptorSet>,
    swapchain_set_layout: Arc<DescriptorSetLayout>,
    viewport: Viewport,
    scissor: Scissor,
    frame_index: usize,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    default_sampler: Arc<Sampler>,

    context_descriptor_set: FrameDescriptorSet,
    frame_uniform_buffers: Vec<Subbuffer<RendererUBO>>,

    mrt_pass: MRT,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();

        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let _debug_callback = unsafe {
            DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo {
                    message_severity: DebugUtilsMessageSeverity::ERROR
                        | DebugUtilsMessageSeverity::WARNING
                        | DebugUtilsMessageSeverity::INFO
                        | DebugUtilsMessageSeverity::VERBOSE,
                    message_type: DebugUtilsMessageType::GENERAL
                        | DebugUtilsMessageType::VALIDATION
                        | DebugUtilsMessageType::PERFORMANCE,
                    ..DebugUtilsMessengerCreateInfo::user_callback(
                        DebugUtilsMessengerCallback::new(
                            |message_severity, message_type, callback_data| {
                                let severity = if message_severity
                                    .intersects(DebugUtilsMessageSeverity::ERROR)
                                {
                                    "error"
                                } else if message_severity
                                    .intersects(DebugUtilsMessageSeverity::WARNING)
                                {
                                    "warning"
                                } else if message_severity
                                    .intersects(DebugUtilsMessageSeverity::INFO)
                                {
                                    "information"
                                } else if message_severity
                                    .intersects(DebugUtilsMessageSeverity::VERBOSE)
                                {
                                    "verbose"
                                } else {
                                    panic!("no-impl");
                                };

                                let ty = if message_type.intersects(DebugUtilsMessageType::GENERAL)
                                {
                                    "general"
                                } else if message_type.intersects(DebugUtilsMessageType::VALIDATION)
                                {
                                    "validation"
                                } else if message_type
                                    .intersects(DebugUtilsMessageType::PERFORMANCE)
                                {
                                    "performance"
                                } else {
                                    panic!("no-impl");
                                };

                                println!(
                                    "{} {} {}: {}",
                                    callback_data.message_id_name.unwrap_or("unknown"),
                                    ty,
                                    severity,
                                    callback_data.message
                                );
                            },
                        ),
                    )
                },
            )
        }
        .ok();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: [QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }]
                .to_vec(),
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },

                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let (models, materials) =
            tobj::load_obj(
                "assets/teapot.obj",
                &tobj::LoadOptions::default()
            )
                .expect("Failed to OBJ load file");

        assert_eq!(models.len(), 1);

        let (models, materials) =
            tobj::load_obj("assets/teapot.obj", &tobj::LoadOptions{
                triangulate: true,
                single_index: true,
                ..Default::default()
            })
                .expect("Failed to OBJ load file");

        let _materials = materials.expect("Failed to load MTL file");
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
                (
                    mesh.texcoords[i * 2 + 0],
                    mesh.texcoords[i * 2 + 1],
                )
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
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE| MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            indices,
        ).unwrap();

        App {
            instance,
            device,
            queue,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            vertex_buffer,
            index_buffer,
            rcx: None,
        }
    }
}

#[derive(BufferContents)]
#[repr(C)]
struct RendererUBO {
    view: [f32; 16],
    proj: [f32; 16],
    sun_direction: [f32; 4],
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0];

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

        let (attachment_image_views, mrt) = window_size_dependent_setup(
            self.device.clone(),
            self.memory_allocator.clone(),
            &images,
        );

        let fullscreen_pipeline = {
            mod vs_fullscreen {
                vulkano_shaders::shader! {
                    ty: "vertex",
                    src: r"
                        #version 450

                        layout(location = 0) out vec2 uvs;

                        void main() {
                            uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                            gl_Position = vec4(uvs * 2.0f + -1.0f, 0.0f, 1.0f);
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

                        layout(set=1, binding=0) uniform sampler2D mrt_pos;
                        layout(set=1, binding=1) uniform sampler2D mrt_normal;

                        layout(set = 0, binding = 0, std140) uniform UBO {
                            mat4 view;
                            mat4 projection;
                            vec4 sun_direction;
                        };

                        void main() {
                            vec3 P = texture(mrt_pos, uvs).xyz;
                            vec3 N = normalize(texture(mrt_normal, uvs).xyz);
                            vec3 L = normalize(sun_direction.xyz);

                            float ndotl = max(dot(N, -L), 0.0);
                            f_color = vec4(vec3(ndotl), 1.0);
                        }
                    ",
                }
            }
            let vs = vs_fullscreen::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs_fullscreen::load(self.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let ubo_layout = {
                let mut binding = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);
                binding.stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

                let mut info = DescriptorSetLayoutCreateInfo::default();
                info.bindings.insert(0, binding);
                info
            };

            let sampler_layout = {
                let mut pos = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);
                pos.descriptor_count = 1;
                pos.stages = ShaderStages::FRAGMENT;

                let mut norm = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);
                norm.descriptor_count = 1;
                norm.stages = ShaderStages::FRAGMENT;


                let mut uvs = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);
                uvs.descriptor_count = 1;
                uvs.stages = ShaderStages::FRAGMENT;

                let mut info = DescriptorSetLayoutCreateInfo::default();
                info.bindings.insert(0, pos);
                info.bindings.insert(1, norm);
                info.bindings.insert(2, uvs);
                info
            };

            let layout = {
                let create_info = PipelineDescriptorSetLayoutCreateInfo {
                    set_layouts: vec![ubo_layout, sampler_layout],
                    flags: PipelineLayoutCreateFlags::empty(),
                    push_constant_ranges: vec![],
                };

                PipelineLayout::new(
                    self.device.clone(),
                    create_info.into_pipeline_layout_create_info(self.device.clone()).unwrap(),
                )
                    .unwrap()
            };

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(Format::B8G8R8A8_UNORM)],
                ..Default::default()
            };

            let mut pi = GraphicsPipelineCreateInfo::layout(layout);
            pi.vertex_input_state = Some(vertex_input_state);
            pi.input_assembly_state = Some(InputAssemblyState::default());
            pi.viewport_state = Some(ViewportState::default());
            pi.rasterization_state = Some(RasterizationState {
                cull_mode: CullMode::Front,
                ..Default::default()
            });
            pi.multisample_state = Some(MultisampleState::default());
            pi.color_blend_state = Some(ColorBlendState {
                attachments: [ColorBlendAttachmentState::default()].to_vec(),
                ..Default::default()
            });
            pi.stages = stages.into_iter().collect();
            pi.subpass = Some((subpass).into());
            pi.dynamic_state.insert(DynamicState::Viewport);
            pi.dynamic_state.insert(DynamicState::Scissor);

            GraphicsPipeline::new(self.device.clone(), None, pi).unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 1.0..=0.0,
        };

        let scissor = Scissor {
            offset: [0, 0],
            extent: window_size.into(),
        };

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        let uniform_buffers = (0..swapchain.image_count())
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

        let default_sampler = Sampler::new(self.device.clone(), SamplerCreateInfo {
                mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_mode: SamplerMipmapMode::Nearest,
            address_mode: [SamplerAddressMode::Repeat, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat],
            mip_lod_bias: 0.0,
            anisotropy: None,
            compare: None,
            lod: 0.0..=1.0,
            border_color: BorderColor::FloatTransparentBlack,
            unnormalized_coordinates: false,
            reduction_mode: SamplerReductionMode::WeightedAverage,
            sampler_ycbcr_conversion: None,
            ..Default::default()
        }).unwrap();

        let (swapchain_set, layout) = {
            let mut pos =
                DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);
            pos.descriptor_count = 1;
            pos.stages = ShaderStages::FRAGMENT;

            let mut norm =
                DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);
            norm.descriptor_count = 1;
            norm.stages = ShaderStages::FRAGMENT;

            let mut uvs =
                DescriptorSetLayoutBinding::descriptor_type(DescriptorType::CombinedImageSampler);
            uvs.descriptor_count = 1;
            uvs.stages = ShaderStages::FRAGMENT;

            let descriptor_layout = DescriptorSetLayout::new(
                self.device.clone(),
                DescriptorSetLayoutCreateInfo {
                    flags: Default::default(),
                    bindings: BTreeMap::from([(0, pos), (1, norm), (2, uvs)]),
                    ..Default::default()
                },
            )
            .unwrap();

            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                descriptor_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        mrt.gbuffer_image_views[1].clone(),
                        default_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        mrt.gbuffer_image_views[0].clone(),
                        default_sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view_sampler(
                        1,
                        mrt.gbuffer_image_views[2].clone(),
                        default_sampler.clone(),
                    ),
                ],
                [],
            )
            .unwrap();

            (set, descriptor_layout)
        };
        self.rcx = Some(RenderContext {
            window,
            swapchain,
            attachment_image_views,
            swapchain_pipeline: fullscreen_pipeline,
            swapchain_set,
            swapchain_set_layout:layout,
            viewport,
            scissor,
            recreate_swapchain,
            frame_index: 0,
            previous_frame_end,
            context_descriptor_set: FrameDescriptorSet::new(
                self.device.clone(),
                self.descriptor_set_allocator.clone(),
                &uniform_buffers,
            ),
            default_sampler,
            frame_uniform_buffers: uniform_buffers,
            mrt_pass: mrt,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                if rcx.recreate_swapchain {
                    let (new_swapchain, new_images) = rcx
                        .swapchain
                        .recreate(SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..rcx.swapchain.create_info()
                        })
                        .expect("failed to recreate swapchain");

                    rcx.swapchain = new_swapchain;

                    (rcx.attachment_image_views, rcx.mrt_pass) = window_size_dependent_setup(
                        self.device.clone(),
                        self.memory_allocator.clone(),
                        &new_images,
                    );

                    rcx.swapchain_set = DescriptorSet::new(
                        self.descriptor_set_allocator.clone(),
                        rcx.swapchain_set_layout.clone(),
                        [
                            WriteDescriptorSet::image_view_sampler(
                                0,
                                rcx.mrt_pass.gbuffer_image_views[1].clone(),
                                rcx.default_sampler.clone(),
                            ),
                            WriteDescriptorSet::image_view_sampler(
                                1,
                                rcx.mrt_pass.gbuffer_image_views[0].clone(),
                                rcx.default_sampler.clone(),
                            ),
                        ],
                        [],
                    ).unwrap();

                    rcx.viewport.extent = window_size.into();
                    rcx.scissor.extent = window_size.into();

                    rcx.recreate_swapchain = false;
                }

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                    rcx.swapchain.clone(),
                    None,
                )
                .map_err(Validated::unwrap)
                {
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

                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                acquire_future.wait(None).unwrap();
                rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

                {
                    const FOV: f32 = 70_f32.to_radians();
                    let mut proj_gl = glm::perspective(window_size.width as f32 / window_size.height as f32, FOV, 0.1, 1000.0);
                    proj_gl[(1,1)] *= -1.0;

                    let eye = TVec3::new(0.0, -3.0, 5.0);
                    let target = TVec3::new(0.0, 0.0, 0.0);
                    let up= TVec3::y().neg();

                    let view = glm::look_at_lh(&eye, &target, &up);

                    let azimuth = 0.8_f32;
                    let phi = 0.6_f32;

                    let sun_dir = Vector3::new(
                        phi.cos() * azimuth.sin(),
                        phi.sin(),
                        phi.cos() * azimuth.cos(),
                    )
                    .normalize();

                    let sun_world = sun_dir;
                    let sun_view = (view * nalgebra::Vector4::new(sun_world.x, sun_world.y, sun_world.z, 0.0))
                        .xyz()
                        .normalize();

                    *rcx.frame_uniform_buffers[image_index as usize]
                        .write()
                        .unwrap() = RendererUBO {
                        view: view.as_slice().try_into().unwrap(),
                        proj: proj_gl.as_slice().try_into().unwrap(),
                        sun_direction: [sun_view.x, sun_view.y, sun_view.z, 0.0],
                    };
                }

                {
                    let model = Matrix4::<f32>::identity()
                        .append_translation(&Vector3::new(0.0, 0.0, 30.0))
                        .scale(0.1f32);
                    let pc = PushConsts {
                        model: model.as_slice().try_into().unwrap(),
                    };

                    builder
                        .begin_rendering(RenderingInfo {
                            render_area_extent: rcx.scissor.extent.clone(),
                            render_area_offset: rcx.scissor.offset.clone(),
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
                                }),Some(RenderingAttachmentInfo {
                                    load_op: AttachmentLoadOp::Clear,
                                    store_op: AttachmentStoreOp::Store,
                                    clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                                    ..RenderingAttachmentInfo::image_view(
                                        rcx.mrt_pass.gbuffer_image_views[2].clone(),
                                    )
                                }),
                            ],
                            depth_attachment:Some(RenderingAttachmentInfo {
                                load_op: AttachmentLoadOp::Clear,
                                store_op: AttachmentStoreOp::Store,
                                clear_value: Some(ClearValue::Depth(0.0)),
                                ..RenderingAttachmentInfo::image_view(
                                    rcx.mrt_pass.gbuffer_image_views[3].clone(),
                                )
                            }),
                            ..Default::default()
                        })
                        .unwrap()
                        .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
                        .unwrap()
                        .set_scissor(0, [rcx.scissor.clone()].into_iter().collect())
                        .unwrap()
                        .bind_pipeline_graphics(rcx.mrt_pass.gbuffer_pipeline.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.mrt_pass.gbuffer_pipeline.layout().clone(),
                            0,
                            rcx.context_descriptor_set
                                .for_frame(rcx.frame_index)
                                .clone(),
                        )
                        .unwrap()
                        .push_constants(rcx.mrt_pass.gbuffer_pipeline.layout().clone(), 0, pc)
                        .unwrap()
                        .bind_vertex_buffers(0, self.vertex_buffer.clone())
                        .unwrap()
                        .bind_index_buffer(self.index_buffer.clone())
                        .unwrap();

                    unsafe { builder.draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0) }.unwrap();

                    builder.end_rendering().unwrap();
                }

                {
                    let descriptor_sets = vec![
                        rcx.context_descriptor_set.for_frame(rcx.frame_index).clone(),
                        rcx.swapchain_set.clone(),
                    ];

                    builder
                        .begin_rendering(RenderingInfo {
                            render_area_extent: rcx.scissor.extent.clone(),
                            render_area_offset: rcx.scissor.offset.clone(),
                            color_attachments: vec![Some(RenderingAttachmentInfo {
                                load_op: AttachmentLoadOp::Clear,
                                store_op: AttachmentStoreOp::Store,
                                clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                                ..RenderingAttachmentInfo::image_view(
                                    rcx.attachment_image_views[image_index as usize].clone(),
                                )
                            })],
                            ..Default::default()
                        })
                        .unwrap()
                        .bind_pipeline_graphics(rcx.swapchain_pipeline.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            rcx.swapchain_pipeline.layout().clone(),
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
                    .then_execute(self.queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            rcx.swapchain.clone(),
                            image_index,
                        ),
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

                let count = rcx.swapchain.image_count() as usize;
                rcx.frame_index = (rcx.frame_index + 1) % count;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    uvs: [f32; 2],
}

fn window_size_dependent_setup(
    device: Arc<Device>,
    allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    images: &[Arc<Image>],
) -> (Vec<Arc<ImageView>>, MRT) {
    let mrt_pipeline = {
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec3 position;
                    layout(location = 1) in vec3 normal;
                    layout(location = 2) in vec2 uvs;

                    layout(location = 0) out vec3 v_position;
                    layout(location = 1) out vec3 v_normal;
                    layout(location = 2) out vec2 v_uvs;

                    layout(set=0, binding=0, std140) uniform UBO {
                        mat4 view;
                        mat4 projection;
                        vec4 sun_direction; // now in view space
                    };

                    layout(push_constant, std430) uniform PC {
                        mat4 model;
                    };

                    void main() {
                        vec4 world_pos  = model * vec4(position, 1.0);
                        mat3 normal_mat = transpose(inverse(mat3(model)));
                        vec3 world_norm = normalize(normal_mat * normal);

                        vec4 view_pos = view * world_pos;
                        v_position = view_pos.xyz;
                        v_normal   = normalize((view * vec4(world_norm, 0.0)).xyz);
                        v_uvs      = uvs;

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

                    layout(location = 0) in vec3 v_position;
                    layout(location = 1) in vec3 v_normals;
                    layout(location = 2) in vec2 v_uvs;

                    layout(location = 0) out vec4 position_mrt;
                    layout(location = 1) out vec4 normal_mrt;
                    layout(location = 2) out vec4 uvs_mrt;

                    void main() {
                        normal_mrt = vec4(normalize(v_normals), 0.0);
                        position_mrt = vec4(v_position, 1.0);
                        uvs_mrt = vec4(v_uvs.xy, 0.0, 0.0);
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

        let ubo_layout = {
            let mut binding = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer);
            binding.stages = ShaderStages::VERTEX | ShaderStages::FRAGMENT;

            let mut info = DescriptorSetLayoutCreateInfo::default();
            info.bindings.insert(0, binding);
            info
        };

        const PUSH_CONSTANT_RANGE: PushConstantRange = PushConstantRange {
                stages: ShaderStages::VERTEX,
            offset: 0,
            size: std::mem::size_of::<vs::PC>() as u32,
        };


        let layout = {
            let create_info = PipelineDescriptorSetLayoutCreateInfo {
                set_layouts: vec![ubo_layout],
                flags: PipelineLayoutCreateFlags::empty(),
                push_constant_ranges: vec![PUSH_CONSTANT_RANGE]
            };

            PipelineLayout::new(
                device.clone(),
                create_info.into_pipeline_layout_create_info(device.clone()).unwrap(),
            )
            .unwrap()
        };

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![
                Some(Format::R16G16B16A16_SFLOAT),
                Some(Format::R16G16B16A16_SFLOAT),
                Some(Format::R16G16B16A16_SFLOAT),
            ],
            depth_attachment_format: Some(Format::D24_UNORM_S8_UINT),
            ..Default::default()
        };

        let mut pi = GraphicsPipelineCreateInfo::layout(layout);
        pi.vertex_input_state = Some(vertex_input_state);
        pi.input_assembly_state = Some(InputAssemblyState::default());
        pi.viewport_state = Some(ViewportState::default());
        pi.rasterization_state = Some(RasterizationState {
            cull_mode: CullMode::Back,
            ..Default::default()
        });
        pi.multisample_state = Some(MultisampleState::default());
        pi.color_blend_state = Some(ColorBlendState {
            attachments: [
                ColorBlendAttachmentState::default(),
                ColorBlendAttachmentState::default(),
                ColorBlendAttachmentState::default(),
            ]
            .to_vec(),
            ..Default::default()
        });
        pi.depth_stencil_state = Some(DepthStencilState {
            depth: Some(DepthState::reverse()),
           ..Default::default()
        });
        pi.stages = stages.into_iter().collect();
        pi.subpass = Some((subpass).into());
        pi.dynamic_state.insert(DynamicState::Viewport);
        pi.dynamic_state.insert(DynamicState::Scissor);

        GraphicsPipeline::new(device.clone(), None, pi).unwrap()
    };

    let window_size = images[0].extent();

    let mut gbuffer_images: Vec<Arc<Image>> = (0..3)
        .map(|_| {
            Image::new(
                allocator.clone(),
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

    gbuffer_images.push(Image::new(
        allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::D24_UNORM_S8_UINT,
            extent: [window_size[0], window_size[1], 1],
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
        .unwrap());
    // Pos, Normal, UV + Depth

    let gbuffer_views: Vec<Arc<ImageView>> = gbuffer_images
        .iter()
        .map(|img| ImageView::new_default(img.clone()).unwrap())
        .collect();

    let mrt = MRT {
        gbuffer_pipeline: mrt_pipeline,
        gbuffer_image_views: gbuffer_views,
    };

    (
        images
            .iter()
            .map(|image| ImageView::new_default(image.clone()).unwrap())
            .collect::<Vec<_>>(),
        mrt,
    )
}
