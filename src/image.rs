use std::sync::Arc;

use vulkano::{buffer::{Buffer, BufferCreateInfo, BufferUsage}, command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryCommandBufferAbstract, allocator::StandardCommandBufferAllocator}, device::{DeviceOwned, Queue}, format::Format, image::{Image, ImageCreateInfo, ImageUsage, max_mip_levels, sampler::Sampler, view::ImageView}, memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, sync::{self, GpuFuture}};

use crate::mesh::ImageViewSampler;

pub struct ImageInfo {
    dimensions: [u32; 2],
    format: Format,
    debug_name: String,
    sampler: Arc<Sampler>,
}

impl ImageInfo {
    pub fn new(
        dimensions: [u32; 2],
        format: Format,
        debug_name: String,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self {
            dimensions,
            format,
            debug_name,
            sampler,
        }
    }

    pub fn white_texture(sampler: Arc<Sampler>) -> Self {
        Self::new(
            [1, 1],
            Format::R8G8B8A8_UNORM,
            "White Texture".to_string(),
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
    let expected_len = (image_info.dimensions[0] as usize)
        * (image_info.dimensions[1] as usize)
        * expected_bytes_per_pixel(image_info.format);

    debug_assert_eq!(
        value.len(),
        expected_len,
        "Image data size does not match format requirements"
    );

    let mip_count = max_mip_levels([image_info.dimensions[0], image_info.dimensions[1], 1]);

    let img = Image::new(
        allocator.clone(),
        ImageCreateInfo {
            format: image_info.format,
            extent: [image_info.dimensions[0], image_info.dimensions[1], 1],
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
        // Perform a clear to the specified value
        let mut builder = AutoCommandBufferBuilder::primary(cb_allocator.clone(), queue.queue_family_index(), CommandBufferUsage::OneTimeSubmit).unwrap();

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

        let ended =  {

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

    Arc::new(ImageViewSampler::new(
        ImageView::new_default(img.clone()).unwrap(),
        image_info.sampler.clone(),
    ))
}
