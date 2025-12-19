use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator,
    },
    device::{DeviceOwned, Queue},
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage, max_mip_levels,
        sampler::{LOD_CLAMP_NONE, Sampler},
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::GpuFuture,
};

use crate::mesh::ImageViewSampler;

#[derive(Clone, Copy, Debug)]
pub enum ImageDimensions {
    Dim2d([u32; 2]),
    Dim3d([u32; 3]),
}

#[derive(Clone)]
pub struct ImageInfo {
    pub dimensions: ImageDimensions,
    pub format: Format,
    pub mips: Option<u32>,
    pub sampler: Arc<Sampler>,
    pub debug_name: String,
}

impl ImageInfo {
    pub fn new(
        dimensions: ImageDimensions,
        format: Format,
        mips: Option<u32>,
        debug_name: String,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self {
            dimensions,
            format,
            mips,
            debug_name,
            sampler,
        }
    }

    pub fn white_texture(sampler: Arc<Sampler>) -> Self {
        Self::new(
            ImageDimensions::Dim2d([1, 1]),
            Format::R8G8B8A8_UNORM,
            None,
            "White Texture".to_string(),
            sampler,
        )
    }

    pub fn black_texture(sampler: Arc<Sampler>) -> Self {
        Self::new(
            ImageDimensions::Dim2d([1, 1]),
            Format::R8G8B8A8_UNORM,
            None,
            "Black Texture".to_string(),
            sampler,
        )
    }
}

fn expected_bytes_per_pixel(format: Format) -> usize {
    match format {
        Format::R8G8B8A8_UNORM => 4,
        _ => unimplemented!("format not supported"),
    }
}

pub fn create_image(
    queue: Arc<Queue>,
    cb_allocator: Arc<StandardCommandBufferAllocator>,
    allocator: Arc<StandardMemoryAllocator>,
    value: &[u8],
    image_info: ImageInfo,
) -> Arc<ImageViewSampler> {
    let (extent, image_type) = match image_info.dimensions {
        ImageDimensions::Dim2d([w, h]) => ([w, h, 1], ImageType::Dim2d),
        ImageDimensions::Dim3d([w, h, d]) => ([w, h, d], ImageType::Dim3d),
    };

    let expected_len = (extent[0] as usize)
        * (extent[1] as usize)
        * (extent[2] as usize)
        * expected_bytes_per_pixel(image_info.format);

    debug_assert_eq!(
        value.len(),
        expected_len,
        "Image data size does not match format requirements"
    );

    let mip_count = if let Some(val) = image_info.mips {
        val.max(1)
    } else {
        max_mip_levels(extent)
    };
    let img = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            image_type,
            format: image_info.format,
            extent,
            mip_levels: mip_count,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();

    allocator
        .device()
        .set_debug_utils_object_name(&img, Some(&image_info.debug_name))
        .unwrap();

    {
        let mut builder = AutoCommandBufferBuilder::primary(
            cb_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

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
            value.iter().copied(),
        )
        .unwrap();

        let ended = {
            builder
                .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(staging, img.clone()))
                .unwrap();

            builder.build().unwrap()
        };

        let result = ended.execute(queue.clone()).unwrap();
        result
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    let view = ImageView::new_default(img.clone()).unwrap();

    Arc::new(ImageViewSampler::new(view, image_info.sampler.clone()))
}
