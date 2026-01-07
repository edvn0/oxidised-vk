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
    RenderSettings,
    bloom_pass::{BloomPass, BloomSettings},
    camera::Camera,
    imgui::{renderer::ImGuiRenderer, settings::Settings},
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
    pub swapchain_pass: SwapchainPass,
    pub bloom_pass: BloomPass,

    pub settings: RenderSettings,

    pub winit_platform: dear_imgui_winit::WinitPlatform,
    pub imgui_context: dear_imgui_rs::Context,
    pub imgui_renderer: ImGuiRenderer,
}

// ============================================================================
// Helper structures for build_frame phases
// ============================================================================

struct InstanceCounts {
    per_submesh: Vec<u32>,
    total: u32,
}

struct InstanceLayout {
    offsets: Vec<u32>,
    total_instances: u32,
}

struct InstanceData {
    transform: TransformTRS,
    material_id: u32,
    submesh_index: usize,
}

// ============================================================================
// Phase 1: Classify draws into pipeline-compatible streams
// ============================================================================

fn classify_draws_into_streams(
    draws: &[DrawSubmission],
    meshes: &MeshRegistry,
) -> HashMap<RenderStreamKey, Vec<usize>> {
    let mut buckets: HashMap<RenderStreamKey, Vec<usize>> = HashMap::new();

    for (index, draw) in draws.iter().enumerate() {
        let mesh = meshes.get(draw.mesh).unwrap();

        let material_class = resolve_material_class(draw, &mesh);

        let key = RenderStreamKey {
            mesh: draw.mesh,
            material_class,
        };

        buckets.entry(key).or_default().push(index);
    }

    buckets
}

fn resolve_material_class(draw: &DrawSubmission, mesh: &MeshAsset) -> MaterialClass {
    if let Some(override_id) = draw.override_material {
        let mat = mesh.materials_buffer.read().unwrap();
        mat.get(override_id as usize)
            .map(|x| x.material_class())
            .unwrap_or(MaterialClass::Opaque)
    } else {
        MaterialClass::Opaque
    }
}

// ============================================================================
// Phase 2: Count instances per submesh
// ============================================================================

fn count_instances_per_submesh(
    draw_indices: &[usize],
    draws: &[DrawSubmission],
    submesh_count: usize,
) -> InstanceCounts {
    let mut counts = vec![0u32; submesh_count];

    for &index in draw_indices {
        increment_counts_for_draw(&draws[index], &mut counts, submesh_count);
    }

    let total = counts.iter().sum();

    InstanceCounts {
        per_submesh: counts,
        total,
    }
}

fn increment_counts_for_draw(draw: &DrawSubmission, counts: &mut [u32], submesh_count: usize) {
    match draw.submesh {
        SubmeshSelection::All => {
            for count in counts.iter_mut() {
                *count += 1;
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

// ============================================================================
// Phase 3: Compute instance layout (prefix sum)
// ============================================================================

fn compute_instance_layout(counts: &InstanceCounts) -> InstanceLayout {
    let mut offsets = vec![0u32; counts.per_submesh.len()];
    let mut running = 0u32;

    for (i, &count) in counts.per_submesh.iter().enumerate() {
        offsets[i] = running;
        running += count;
    }

    InstanceLayout {
        offsets,
        total_instances: running,
    }
}

// ============================================================================
// Phase 4: Patch indirect draw commands
// ============================================================================

fn patch_indirect_commands(
    indirect_buffer: &Subbuffer<[DrawIndexedIndirectCommand]>,
    layout: &InstanceLayout,
    counts: &InstanceCounts,
) {
    if let Ok(mut cmds) = indirect_buffer.write() {
        for i in 0..counts.per_submesh.len() {
            cmds[i].first_instance = layout.offsets[i];
            cmds[i].instance_count = counts.per_submesh[i];
        }
    }
}

// ============================================================================
// Phase 5: Flatten draws into instance data
// ============================================================================

fn flatten_draws_to_instances(
    draw_indices: &[usize],
    draws: &[DrawSubmission],
    mesh: &MeshAsset,
    submesh_count: usize,
) -> Vec<InstanceData> {
    let mut instances = Vec::new();

    for &index in draw_indices {
        let draw = &draws[index];
        match draw.submesh {
            SubmeshSelection::All => {
                for si in 0..submesh_count {
                    instances.push(create_instance_data(draw, mesh, si));
                }
            }
            SubmeshSelection::One(i) => {
                let si = i as usize;
                if si < submesh_count {
                    instances.push(create_instance_data(draw, mesh, si));
                }
            }
        }
    }

    instances
}

fn create_instance_data(
    draw: &DrawSubmission,
    mesh: &MeshAsset,
    submesh_index: usize,
) -> InstanceData {
    let material_id = resolve_material(mesh, submesh_index, draw.override_material);

    InstanceData {
        transform: draw.transform,
        material_id,
        submesh_index,
    }
}

fn resolve_material(mesh: &MeshAsset, submesh_index: usize, override_material: Option<u32>) -> u32 {
    override_material.unwrap_or(mesh.submeshes[submesh_index].material_index)
}

// ============================================================================
// Phase 6: Write instance data to buffers
// ============================================================================

fn write_instance_data_to_buffers(
    instances: &[InstanceData],
    layout: &InstanceLayout,
    transforms_buffer: &Subbuffer<[TransformTRS]>,
    material_ids_buffer: &Subbuffer<[u32]>,
    submesh_count: usize,
) {
    let mut heads = vec![0u32; submesh_count];

    if let (Ok(mut transforms), Ok(mut material_ids)) =
        (transforms_buffer.write(), material_ids_buffer.write())
    {
        for instance in instances {
            let si = instance.submesh_index;
            let dst = (layout.offsets[si] + heads[si]) as usize;
            heads[si] += 1;

            transforms[dst] = instance.transform;
            material_ids[dst] = instance.material_id;
        }
    }
}

// ============================================================================
// Main entry point: Refactored build_frame
// ============================================================================

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
        let stream_buckets = classify_draws_into_streams(&self.frame_submission.draws, &meshes);

        // Drop meshes lock before processing streams (which needs mutable access to self)
        drop(meshes);

        /* ---------------------------------------------------------
         * Phase 2-6: build per-stream instance data
         *
         * Each stream:
         *   - owns all submeshes of the mesh
         *   - has one indirect draw command per submesh
         *   - packs instances contiguously per submesh
         *
         * Indirect draw commands are *patched*, never recreated.
         * --------------------------------------------------------- */
        for (key, draw_indices) in stream_buckets {
            self.process_render_stream(&key, &draw_indices, &allocator);
        }
    }

    fn process_render_stream(
        &mut self,
        key: &RenderStreamKey,
        draw_indices: &[usize],
        allocator: &Arc<StandardMemoryAllocator>,
    ) {
        let stream = self.mesh_streams.get_mut(key).unwrap();
        let submesh_count = stream.mesh.submeshes.len();

        /* ---- Phase 2: count instances per submesh ----
         *
         * This determines instance ranges for each indirect draw.
         * We intentionally count first to avoid branching while writing.
         */
        let counts =
            count_instances_per_submesh(draw_indices, &self.frame_submission.draws, submesh_count);

        /* ---- Phase 3: prefix sum → first_instance ----
         *
         * This creates a stable layout:
         *   [submesh0 instances][submesh1 instances][...]
         *
         * This layout is relied upon by:
         *   - indirect draws
         *   - instance indexing in shaders
         */
        let layout = compute_instance_layout(&counts);

        stream.instance_count = layout.total_instances;

        /* ---- Phase 4: ensure instance buffer capacity ----
         *
         * Allocation policy is centralized inside MeshDrawStream.
         * build_frame must remain allocation-agnostic.
         */
        stream.ensure_instance_capacity(allocator, layout.total_instances as usize);

        /* ---- Phase 5: patch indirect commands ----
         *
         * Each indirect command corresponds to exactly one submesh.
         * first_index / index_count are static and never touched here.
         */
        patch_indirect_commands(&stream.indirect, &layout, &counts);

        /* ---- Phase 6: flatten and write instance data ----
         *
         * Heads track the current write position per submesh.
         * No bounds checks are required because counts were precomputed.
         */
        let instances = flatten_draws_to_instances(
            draw_indices,
            &self.frame_submission.draws,
            &stream.mesh,
            submesh_count,
        );

        write_instance_data_to_buffers(
            &instances,
            &layout,
            &stream.transforms,
            &stream.material_ids,
            submesh_count,
        );
    }
}

// ============================================================================
// Unit tests for individual phases
// ============================================================================

#[cfg(test)]
mod tests {
    use gltf::json::extensions::mesh::Mesh;

    use crate::mesh_registry::MeshHandle;

    use super::*;

    #[test]
    fn test_count_instances_single_submesh() {
        // Test counting when all draws target the same submesh
        let draws = vec![
            DrawSubmission {
                submesh: SubmeshSelection::One(0),
                mesh: MeshHandle(0),
                transform: [0.0; 16],
                override_material: None,
            },
            DrawSubmission {
                submesh: SubmeshSelection::One(0),
                mesh: MeshHandle(0),
                transform: [0.0; 16],
                override_material: None,
            },
        ];

        let indices = vec![0, 1];
        let counts = count_instances_per_submesh(&indices, &draws, 3);

        assert_eq!(counts.per_submesh, vec![2, 0, 0]);
        assert_eq!(counts.total, 2);
    }

    #[test]
    fn test_count_instances_all_submeshes() {
        // Test counting when draw targets all submeshes
        let draws = vec![DrawSubmission {
            submesh: SubmeshSelection::All,
            mesh: MeshHandle(0),
            transform: [0.0; 16],
            override_material: None,
        }];

        let indices = vec![0];
        let counts = count_instances_per_submesh(&indices, &draws, 3);

        assert_eq!(counts.per_submesh, vec![1, 1, 1]);
        assert_eq!(counts.total, 3);
    }

    #[test]
    fn test_compute_instance_layout() {
        // Test prefix sum calculation for instance offsets
        let counts = InstanceCounts {
            per_submesh: vec![2, 3, 1],
            total: 6,
        };

        let layout = compute_instance_layout(&counts);

        assert_eq!(layout.offsets, vec![0, 2, 5]);
        assert_eq!(layout.total_instances, 6);
    }

    #[test]
    fn test_instance_layout_empty() {
        // Test edge case: no instances
        let counts = InstanceCounts {
            per_submesh: vec![0, 0, 0],
            total: 0,
        };

        let layout = compute_instance_layout(&counts);

        assert_eq!(layout.offsets, vec![0, 0, 0]);
        assert_eq!(layout.total_instances, 0);
    }
}
