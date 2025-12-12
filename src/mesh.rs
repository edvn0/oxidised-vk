use crate::texture_cache::{BindlessTextureIndex, TextureCache};
use crate::vertex::{PositionMeshVertex, StandardMeshVertex};
use gltf::json::extensions::mesh::Mesh;
use meshopt_rs::vertex::Position;
use meshopt_rs::{
    overdraw::optimize_overdraw,
    simplify,
    vertex::{cache::optimize_vertex_cache, fetch::optimize_vertex_fetch},
};
use nalgebra::{Matrix4, Vector3, Vector4};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator,
    },
    device::Queue,
    format::Format,
    image::{Image, ImageCreateInfo, ImageLayout, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
};

#[derive(Debug)]
pub enum MeshLoadError {
    Io,
    Gltf,
    UnsupportedFormat,
    MissingPositions,
}

impl From<gltf::Error> for MeshLoadError {
    fn from(_: gltf::Error) -> Self {
        Self::Gltf
    }
}

impl From<std::io::Error> for MeshLoadError {
    fn from(_: std::io::Error) -> Self {
        Self::Io
    }
}

#[repr(C)]
#[derive(BufferContents, Clone, Copy, Default)]
pub struct GpuMaterial {
    pub base_color: [f32; 4],

    pub metallic: f32,
    pub roughness: f32,
    pub base_color_tex: u32,
    pub normal_tex: u32,

    pub metallic_roughness_tex: u32,
    pub ao_tex: u32,
    pub _padding: [f32; 2],

    pub flags: u32,

    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

pub struct MeshAsset {
    pub vertex_buffer: Subbuffer<[StandardMeshVertex]>,
    pub position_vertex_buffer: Subbuffer<[PositionMeshVertex]>,
    pub index_buffer: Subbuffer<[u32]>,

    pub submeshes: Vec<SubmeshGpu>,
    pub lods: Vec<MeshLod>,

    pub materials_buffer: Subbuffer<[GpuMaterial]>,
    pub texture_array: Vec<Arc<ImageView>>,
}

impl MeshAsset {
    pub fn index_count(&self) -> u32 {
        self.index_buffer.len() as u32
    }
}

#[derive(Clone, Copy)]
pub struct SubmeshGpu {
    pub first_index: u32,
    pub index_count: u32,
    pub material_index: u32,
}

#[derive(Clone, Copy)]
pub struct MeshLod {
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

pub struct BuildMesh {
    pub vertices: Vec<StandardMeshVertex>,
    pub lod_indices: Vec<Vec<u32>>,
    pub submeshes: Vec<SubmeshCpu>,
    pub materials: Vec<MaterialAsset>,
    pub generate_lods: bool,
}

pub struct SubmeshCpu {
    pub first_index: u32,
    pub index_count: u32,
    pub material_index: u32,
}

pub struct MaterialAsset {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,

    pub base_colour_tex: BindlessTextureIndex,
    pub normal_tex: BindlessTextureIndex,
    pub metallic_roughness_tex: BindlessTextureIndex,
    pub ao_tex: BindlessTextureIndex,

    pub flags: u32,
}

impl MaterialAsset {
    pub fn new(
        base_color: [f32; 3],
        metallic: f32,
        roughness: f32,
        base_colour_tex: BindlessTextureIndex,
        normal_tex: BindlessTextureIndex,
        metallic_roughness_tex: BindlessTextureIndex,
        ao_tex: BindlessTextureIndex,
    ) -> Self {
        let has = |t: BindlessTextureIndex| (t.raw() != NON_DEFAULT_TEXTURE_WHITE) as u32;

        let flags = has(base_colour_tex) << 0
            | has(normal_tex) << 1
            | has(metallic_roughness_tex) << 2
            | has(ao_tex) << 3;

        MaterialAsset {
            base_color,
            metallic,
            roughness,
            base_colour_tex,
            normal_tex,
            metallic_roughness_tex,
            ao_tex,
            flags,
        }
    }
}

pub fn upload_build_mesh(
    build: BuildMesh,
    texture_array: Vec<Arc<ImageView>>,
    allocator: &Arc<StandardMemoryAllocator>,
) -> Result<Arc<MeshAsset>, MeshLoadError> {
    assert!(!build.vertices.is_empty());
    assert!(!build.submeshes.is_empty());

    let vb = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        build.vertices.clone(),
    )
    .unwrap();

    let mut flat_buffer = Vec::with_capacity(build.vertices.len() * 3);
    for vertex in &build.vertices {
        flat_buffer.push(PositionMeshVertex::new(vertex.pos()));
    }
    let svb = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        flat_buffer,
    )
    .unwrap();

    let mut merged_indices = Vec::new();
    let mut lods_gpu = Vec::new();

    let base = &build.lod_indices[0];
    merged_indices.extend_from_slice(base);

    lods_gpu.push(MeshLod {
        first_index: 0,
        index_count: base.len() as u32,
        vertex_offset: 0,
    });

    let ib = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        merged_indices,
    )
    .unwrap();

    #[inline]
    const fn extend3(v: [f32; 3], w: f32) -> [f32; 4] {
        [v[0], v[1], v[2], w]
    }

    let gpu_materials = build
        .materials
        .iter()
        .map(|m| GpuMaterial {
            base_color: extend3(m.base_color, 1.0),
            metallic: m.metallic,
            roughness: m.roughness,
            base_color_tex: m.base_colour_tex.raw(),
            normal_tex: m.normal_tex.raw(),
            metallic_roughness_tex: m.metallic_roughness_tex.raw(),
            ao_tex: m.ao_tex.raw(),
            flags: m.flags,
            ..Default::default()
        })
        .collect::<Vec<_>>();

    let materials_buffer = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        gpu_materials,
    )
    .unwrap();

    Ok(Arc::new(MeshAsset {
        vertex_buffer: vb,
        position_vertex_buffer: svb,
        index_buffer: ib,

        submeshes: build
            .submeshes
            .iter()
            .map(|s| SubmeshGpu {
                first_index: s.first_index,
                index_count: s.index_count,
                material_index: s.material_index,
            })
            .collect(),

        lods: lods_gpu,

        texture_array,
        materials_buffer,
    }))
}

pub fn import_gltf(
    path: &str,
    allocator: &Arc<StandardMemoryAllocator>,
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
    white_texture: &Arc<ImageView>,
    generate_lods: bool,
) -> Result<(BuildMesh, Vec<Arc<ImageView>>), MeshLoadError> {
    let (gltf, buffers, images) = gltf::import(path)?;

    let mut texture_array = Vec::new();
    let mut texture_cache = TextureCache::new(BindlessTextureIndex::WHITE);

    // index 0 = white texture
    texture_array.push(white_texture.clone());

    // real textures start at 1
    for (i, img) in images.iter().enumerate() {
        let view = upload_gltf_image(img, allocator, cb_allocator, queue)?;
        let gpu_index = BindlessTextureIndex((1 + i) as u32);
        texture_cache.insert(i, gpu_index);
        texture_array.push(view);
    }

    let mut materials = Vec::new();
    for mat in gltf.materials() {
        materials.push(build_gltf_material_cpu(&mat, &texture_cache));
    }

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut submeshes = Vec::new();

    let identity = Matrix4::<f32>::identity();

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            process_node(
                &node,
                &identity,
                &mut vertices,
                &mut indices,
                &mut submeshes,
                &buffers,
            )?;
        }
    }

    let lods = generate_lods_meshopt(&vertices, &indices, &[0.75, 0.5, 0.25]);

    Ok((
        BuildMesh {
            vertices,
            lod_indices: lods,
            submeshes,
            materials,
            generate_lods,
        },
        texture_array,
    ))
}

const NON_DEFAULT_TEXTURE_WHITE: u32 = 0; // This is the white texture

fn build_gltf_material_cpu(material: &gltf::Material, cache: &TextureCache) -> MaterialAsset {
    let pbr = material.pbr_metallic_roughness();

    MaterialAsset::new(
        pbr.base_color_factor()[0..3].try_into().unwrap(),
        pbr.metallic_factor(),
        pbr.roughness_factor(),
        cache.from_opt_info(pbr.base_color_texture()),
        cache.from_normal_texture(material.normal_texture()),
        cache.from_opt_info(pbr.metallic_roughness_texture()),
        cache.from_ao_texture(material.occlusion_texture()),
    )
}

fn upload_gltf_image(
    img: &gltf::image::Data,
    allocator: &Arc<StandardMemoryAllocator>,
    cb_allocator: &Arc<StandardCommandBufferAllocator>,
    queue: &Arc<Queue>,
) -> Result<Arc<ImageView>, MeshLoadError> {
    let (pixels, format) = match img.format {
        gltf::image::Format::R8G8B8A8 => (img.pixels.clone(), Format::R8G8B8A8_SRGB),

        gltf::image::Format::R8G8B8 => {
            let mut rgba = Vec::with_capacity((img.width * img.height * 4) as usize);
            for rgb in img.pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
            (rgba, Format::R8G8B8A8_SRGB)
        }

        _ => return Err(MeshLoadError::UnsupportedFormat),
    };

    let staging = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        pixels,
    )
    .unwrap();

    let image = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            extent: [img.width, img.height, 1],
            format,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            initial_layout: ImageLayout::Undefined,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let mut cb = AutoCommandBufferBuilder::primary(
        cb_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    cb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, image.clone()))
        .unwrap();

    cb.build()
        .unwrap()
        .execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    Ok(ImageView::new_default(image).unwrap())
}

pub fn generate_lods_meshopt(
    vertices: &[StandardMeshVertex],
    indices: &[u32],
    ratios: &[f32],
) -> Vec<Vec<u32>> {
    let mut lods = Vec::with_capacity(ratios.len() + 1);

    let mut base = indices.to_vec();
    lods.push(base.clone());

    for &ratio in ratios {
        let target_index_count = ((base.len() as f32 * ratio) as usize / 3) * 3;

        let mut dst = vec![0u32; base.len()];

        let size = simplify::simplify(&mut dst, &base, vertices, target_index_count, 0.01);

        dst.truncate(size);

        let cache_copy = dst.clone();
        optimize_vertex_cache(&mut dst, &cache_copy, vertices.len());

        let overdraw_copy = dst.clone();
        optimize_overdraw(&mut dst, &overdraw_copy, vertices, 1.01);

        let mut vtx_copy = vertices.to_vec();
        optimize_vertex_fetch(&mut vtx_copy, &mut dst, vertices);

        lods.push(dst.clone());
        base = dst;
    }

    lods
}

fn process_node(
    node: &gltf::Node,
    parent_world: &Matrix4<f32>,
    vertices: &mut Vec<StandardMeshVertex>,
    indices: &mut Vec<u32>,
    submeshes: &mut Vec<SubmeshCpu>,
    buffers: &[gltf::buffer::Data],
) -> Result<(), MeshLoadError> {
    let local = gltf_matrix_to_na(&node.transform().matrix());
    let world = parent_world * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| Some(&buffers[b.index()]));

            let positions = reader
                .read_positions()
                .ok_or(MeshLoadError::MissingPositions)?;
            let normals = reader.read_normals();
            let uvs = reader.read_tex_coords(0);

            let base_vertex = vertices.len() as u32;

            let normal_matrix = world
                .fixed_view::<3, 3>(0, 0)
                .try_inverse()
                .unwrap_or_else(|| nalgebra::Matrix3::identity())
                .transpose();

            for (i, p) in positions.enumerate() {
                let pos = world * Vector4::new(p[0], p[1], p[2], 1.0);

                let n = normals
                    .as_ref()
                    .and_then(|n| n.clone().nth(i))
                    .unwrap_or([0.0, 0.0, 0.0]);

                let n = (normal_matrix * Vector3::new(n[0], n[1], n[2])).normalize();

                let uv = uvs
                    .as_ref()
                    .and_then(|t| t.clone().into_f32().nth(i))
                    .unwrap_or([0.0, 0.0]);

                vertices.push(StandardMeshVertex::new(
                    [pos.x, pos.y, pos.z],
                    [n.x, n.y, n.z],
                    uv,
                ));
            }

            let first_index = indices.len() as u32;

            if let Some(idx) = reader.read_indices() {
                match idx {
                    gltf::mesh::util::ReadIndices::U8(it) => {
                        indices.extend(it.map(|i| i as u32 + base_vertex))
                    }
                    gltf::mesh::util::ReadIndices::U16(it) => {
                        indices.extend(it.map(|i| i as u32 + base_vertex))
                    }
                    gltf::mesh::util::ReadIndices::U32(it) => {
                        indices.extend(it.map(|i| i + base_vertex))
                    }
                }
            } else {
                let count = vertices.len() as u32 - base_vertex;
                indices.extend((0..count).map(|i| i + base_vertex));
            }

            let index_count = indices.len() as u32 - first_index;

            submeshes.push(SubmeshCpu {
                first_index,
                index_count,
                material_index: primitive.material().index().unwrap_or(0) as u32,
            });
        }
    }

    for child in node.children() {
        process_node(&child, &world, vertices, indices, submeshes, buffers)?;
    }

    Ok(())
}

fn gltf_matrix_to_na(m: &[[f32; 4]; 4]) -> Matrix4<f32> {
    Matrix4::from_columns(&[
        Vector4::new(m[0][0], m[0][1], m[0][2], m[0][3]),
        Vector4::new(m[1][0], m[1][1], m[1][2], m[1][3]),
        Vector4::new(m[2][0], m[2][1], m[2][2], m[2][3]),
        Vector4::new(m[3][0], m[3][1], m[3][2], m[3][3]),
    ])
}
