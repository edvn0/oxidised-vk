use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};

use imgui::Context;
use imgui_winit_support::WinitPlatform;
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::DrawIndexedIndirectCommand,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::{
        rasterization::{CullMode, FrontFace},
        viewport::{Scissor, Viewport},
    },
    swapchain::Swapchain,
    sync::GpuFuture,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    MAX_FRAMES_IN_FLIGHT,
    bloom_pass::{BloomPass, BloomSettings},
    camera::Camera,
    imgui::renderer::ImGuiRenderer,
    main_helpers::FrameDescriptorSet,
    mesh::{ImageViewSampler, MeshAsset},
    mesh_registry::{MeshHandle, MeshRegistry},
    render_passes::recordings::{Composite, CompositeSettings, MRT, MRTLighting, SwapchainPass},
    shader_bindings::RendererUBO,
    submission::{DrawSubmission, FrameSubmission},
};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Culling {
    None,
    Back,
    Front,
    All,
}

impl Culling {
    pub fn next(self) -> Self {
        match self {
            Self::None => Self::Back,
            Self::Back => Self::Front,
            Self::Front => Self::All,
            Self::All => Self::None,
        }
    }
}

impl From<Culling> for CullMode {
    fn from(value: Culling) -> Self {
        match value {
            Culling::None => CullMode::None,
            Culling::Back => CullMode::Back,
            Culling::Front => CullMode::Front,
            Culling::All => CullMode::FrontAndBack,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Winding {
    CounterClockwise,
    Clockwise,
}

impl Winding {
    pub fn toggle(self) -> Self {
        match self {
            Self::CounterClockwise => Self::Clockwise,
            Self::Clockwise => Self::CounterClockwise,
        }
    }
}

impl From<Winding> for FrontFace {
    fn from(value: Winding) -> Self {
        match value {
            Winding::Clockwise => FrontFace::Clockwise,
            Winding::CounterClockwise => FrontFace::CounterClockwise,
        }
    }
}

pub type TransformTRS = [f32; 16];

pub struct MeshDrawStream {
    pub mesh: Arc<MeshAsset>,
    pub instance_count: u32,
    pub material_ids: Subbuffer<[u32]>,
    pub indirect: [Subbuffer<[DrawIndexedIndirectCommand]>; MAX_FRAMES_IN_FLIGHT],
    pub transforms: [Subbuffer<[TransformTRS]>; MAX_FRAMES_IN_FLIGHT],
}

impl MeshDrawStream {
    pub fn new(
        mesh: Arc<MeshAsset>,
        material_ids: Subbuffer<[u32]>,
        indirect: [Subbuffer<[DrawIndexedIndirectCommand]>; MAX_FRAMES_IN_FLIGHT],
        transforms: [Subbuffer<[TransformTRS]>; MAX_FRAMES_IN_FLIGHT],
    ) -> Self {
        Self {
            mesh,
            material_ids,
            indirect,
            transforms,
            instance_count: 0,
        }
    }
}

pub struct FrameResources {
    uniform_buffer: Subbuffer<RendererUBO>,
}

impl FrameResources {
    pub fn new(buf: &Subbuffer<RendererUBO>) -> Self {
        Self {
            uniform_buffer: buf.clone(),
        }
    }
}

pub struct RenderContext {
    pub window: Arc<Window>,
    pub swapchain: Arc<Swapchain>,
    pub viewport: Viewport,
    pub scissor: Scissor,
    pub recreate_swapchain: bool,
    pub previous_frame_end: Option<Box<dyn GpuFuture>>,
    pub start_time: Instant,
    pub elapsed_millis: u64,

    pub meshes: Arc<RwLock<MeshRegistry>>,
    pub mesh_streams: HashMap<MeshHandle, MeshDrawStream>,

    pub white_image_sampler: Arc<ImageViewSampler>,
    pub black_image_sampler: Arc<ImageViewSampler>,

    pub context_descriptor_set: FrameDescriptorSet,

    pub frame_submission: FrameSubmission,
    pub current_frame: usize,
    pub frames: Vec<FrameResources>,

    pub mrt_pass: MRT,
    pub mrt_lighting: MRTLighting,

    pub composite: Composite,
    pub composite_settings: CompositeSettings,

    pub swapchain_pass: SwapchainPass,

    pub bloom_pass: BloomPass,
    pub bloom_settings: BloomSettings,

    pub winit_platform: WinitPlatform,
    pub imgui_context: Context,
    pub imgui_renderer: ImGuiRenderer,
}

impl RenderContext {
    pub fn update_camera_ubo(
        &self,
        camera: &Camera,
        current_frame: usize,
        window_size: PhysicalSize<u32>,
    ) {
        let aspect = window_size.width as f32 / window_size.height as f32;

        let view = camera.view_matrix();
        let proj = camera.projection_matrix(aspect);
        let inverse_proj = proj.try_inverse().unwrap();

        let sun = camera.sun_direction_view_space();

        if let Ok(mut w) = self.frames[current_frame].uniform_buffer.write() {
            *w = RendererUBO {
                view: view.as_slice().try_into().unwrap(),
                proj: proj.as_slice().try_into().unwrap(),
                inverse_proj: inverse_proj.as_slice().try_into().unwrap(),
                sun_direction: sun,
            };
        }
    }

    pub fn build_frame(&mut self, allocator: Arc<StandardMemoryAllocator>) {
        let submission = &self.frame_submission;

        let mut buckets: HashMap<MeshHandle, Vec<&DrawSubmission>> = HashMap::new();
        for draw in &submission.draws {
            buckets.entry(draw.mesh).or_default().push(draw);
        }

        for (mesh_handle, draws) in buckets {
            let stream = self.mesh_streams.get_mut(&mesh_handle).unwrap();

            let required = draws.len();

            let current = &stream.transforms[self.current_frame];
            if current.len() < required as u64 {
                let new_capacity = required
                    .checked_next_power_of_two()
                    .unwrap_or(required as usize)
                    .max(1);

                for slot in &mut stream.transforms {
                    *slot = Buffer::new_slice::<TransformTRS>(
                        allocator.clone(),
                        BufferCreateInfo {
                            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                            ..Default::default()
                        },
                        AllocationCreateInfo {
                            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                            ..Default::default()
                        },
                        new_capacity as DeviceSize,
                    )
                    .unwrap();
                }
            }

            if let Ok(mut w) = stream.transforms[self.current_frame].write() {
                for (i, d) in draws.iter().enumerate() {
                    w[i] = d.transform;
                }
            }

            stream.instance_count = required as u32;

            if let Ok(mut cmds) = stream.indirect[self.current_frame].write() {
                for cmd in cmds.iter_mut() {
                    cmd.first_instance = 0;
                    cmd.instance_count = stream.instance_count;
                }
            }
        }
    }
}
