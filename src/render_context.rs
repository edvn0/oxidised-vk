use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Instant,
};

use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
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
    bloom_pass::{BloomPass, BloomSettings},
    camera::Camera,
    imgui::renderer::ImGuiRenderer,
    main_helpers::FrameDescriptorSet,
    mesh::{ImageViewSampler, MaterialClass, MeshAsset},
    mesh_registry::{MeshRegistry, RenderStreamKey},
    render_passes::recordings::{Composite, CompositeSettings, MRT, MRTLighting, SwapchainPass},
    shader_bindings::RendererUBO,
    submission::{DrawSubmission, FrameSubmission, SubmeshSelection},
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
    pub indirect: Subbuffer<[DrawIndexedIndirectCommand]>,
    pub transforms: Subbuffer<[TransformTRS]>,
}

impl MeshDrawStream {
    pub fn new(
        mesh: Arc<MeshAsset>,
        material_ids: Subbuffer<[u32]>,
        indirect: Subbuffer<[DrawIndexedIndirectCommand]>,
        transforms: Subbuffer<[TransformTRS]>,
    ) -> Self {
        Self {
            mesh,
            material_ids,
            indirect,
            transforms,
            instance_count: 0,
        }
    }

    pub fn ensure_instance_capacity(
        &mut self,
        allocator: &Arc<StandardMemoryAllocator>,
        required_instances: usize,
    ) {
        let current = self.transforms.len() as usize;
        if current >= required_instances {
            return;
        }

        let new_capacity = required_instances
            .checked_next_power_of_two()
            .unwrap_or(required_instances)
            .max(1);

        self.transforms = Buffer::new_slice::<TransformTRS>(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
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

        self.material_ids = Buffer::new_slice::<u32>(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::SHADER_DEVICE_ADDRESS,
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
    pub mesh_streams: HashMap<RenderStreamKey, MeshDrawStream>,

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

    pub winit_platform: dear_imgui_winit::WinitPlatform,
    pub imgui_context: dear_imgui_rs::Context,
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
        let meshes = self.meshes.read().unwrap();

        /* ---------------------------------------------------------
         * Phase 1: classify submissions into pipeline-compatible streams
         *
         * Invariant:
         *   Every stream represents a (mesh × material_class) combination
         *   that can be drawn with a single pipeline configuration.
         *
         * ECS is intentionally unaware of this classification.
         * --------------------------------------------------------- */

        let mut buckets: HashMap<RenderStreamKey, Vec<&DrawSubmission>> = HashMap::new();

        for draw in &submission.draws {
            let mesh = meshes.get(draw.mesh).unwrap();

            // Material *behavior* (not data) determines pipeline compatibility.
            // Overrides may change material data, but must not silently change class.
            let material_class = if let Some(override_id) = draw.override_material {
                let mat = mesh.materials_buffer.read().unwrap()[override_id as usize];
                mat.material_class()
            } else {
                MaterialClass::Opaque
            };

            let key = RenderStreamKey {
                mesh: draw.mesh,
                material_class,
            };

            buckets.entry(key).or_default().push(draw);
        }

        /* ---------------------------------------------------------
         * Phase 2: build per-stream instance data
         *
         * Each stream:
         *   - owns all submeshes of the mesh
         *   - has one indirect draw command per submesh
         *   - packs instances contiguously per submesh
         *
         * Indirect draw commands are *patched*, never recreated.
         * --------------------------------------------------------- */

        for (key, draws) in buckets {
            let stream = self.mesh_streams.get_mut(&key).unwrap();
            let submesh_count = stream.mesh.submeshes.len();

            /* ---- count instances per submesh ----
             *
             * This determines instance ranges for each indirect draw.
             * We intentionally count first to avoid branching while writing.
             */

            let mut counts = vec![0u32; submesh_count];

            for d in &draws {
                match d.submesh {
                    SubmeshSelection::All => {
                        for c in &mut counts {
                            *c += 1;
                        }
                    }
                    SubmeshSelection::One(i) => {
                        let si = i as usize;
                        if si < submesh_count {
                            counts[si] += 1;
                        }
                    }
                }
            }

            /* ---- prefix sum → first_instance ----
             *
             * This creates a stable layout:
             *   [submesh0 instances][submesh1 instances][...]
             *
             * This layout is relied upon by:
             *   - indirect draws
             *   - instance indexing in shaders
             */

            let mut offsets = vec![0u32; submesh_count];
            let mut running = 0u32;

            for i in 0..submesh_count {
                offsets[i] = running;
                running += counts[i];
            }

            stream.instance_count = running;
            let total_instances = running as usize;

            /* ---- ensure instance buffer capacity ----
             *
             * Allocation policy is centralized inside MeshDrawStream.
             * build_frame must remain allocation-agnostic.
             */

            stream.ensure_instance_capacity(&allocator, total_instances);

            /* ---- patch indirect commands ----
             *
             * Each indirect command corresponds to exactly one submesh.
             * first_index / index_count are static and never touched here.
             */

            if let Ok(mut cmds) = stream.indirect.write() {
                for i in 0..submesh_count {
                    cmds[i].first_instance = offsets[i];
                    cmds[i].instance_count = counts[i];
                }
            }

            /* ---- write instance data ----
             *
             * Heads track the current write position per submesh.
             * No bounds checks are required because counts were precomputed.
             */

            let mut heads = vec![0u32; submesh_count];

            if let (Ok(mut t), Ok(mut m)) = (stream.transforms.write(), stream.material_ids.write())
            {
                for d in &draws {
                    let mat_id = d.override_material.unwrap_or(0);

                    match d.submesh {
                        SubmeshSelection::All => {
                            for si in 0..submesh_count {
                                let dst = (offsets[si] + heads[si]) as usize;
                                heads[si] += 1;

                                t[dst] = d.transform;
                                m[dst] = mat_id;
                            }
                        }
                        SubmeshSelection::One(i) => {
                            let si = i as usize;
                            if si < submesh_count {
                                let dst = (offsets[si] + heads[si]) as usize;
                                heads[si] += 1;

                                t[dst] = d.transform;
                                m[dst] = mat_id;
                            }
                        }
                    }
                }
            }
        }
    }
}
