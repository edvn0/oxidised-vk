use crate::shader_bindings::{RendererUBO, renderer_set_0_layouts};
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;

pub struct FrameDescriptorSet {
    sets: Vec<Arc<DescriptorSet>>,
}

impl FrameDescriptorSet {
    pub fn new(
        device: Arc<Device>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        uniform_buffers: &[Subbuffer<RendererUBO>],
    ) -> Self {
        let set_0 = renderer_set_0_layouts();
        let layout = DescriptorSetLayout::new(device.clone(), set_0[0usize].clone()).unwrap();

        let sets: Vec<_> = uniform_buffers
            .iter()
            .map(|ub| {
                let vec = vec![WriteDescriptorSet::buffer(0, ub.clone())];
                DescriptorSet::new(descriptor_set_allocator.clone(), layout.clone(), vec, [])
                    .unwrap()
            })
            .collect();

        Self { sets }
    }

    pub fn for_frame(&self, index: usize) -> &Arc<DescriptorSet> {
        &self.sets[index]
    }
}

pub fn generate_identity_lut_3d(size: u32) -> Vec<u8> {
    let mut data = Vec::with_capacity((size * size * size * 4) as usize);

    let max = (size - 1) as f32;

    for b in 0..size {
        for g in 0..size {
            for r in 0..size {
                let rf = r as f32 / max;
                let gf = g as f32 / max;
                let bf = b as f32 / max;

                data.push((rf * 255.0).round() as u8);
                data.push((gf * 255.0).round() as u8);
                data.push((bf * 255.0).round() as u8);
                data.push(255);
            }
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lut_3d(data: &[u8], size: u32, r: f32, g: f32, b: f32) -> (u8, u8, u8, u8) {
        let r = r.clamp(0.0, 1.0);
        let g = g.clamp(0.0, 1.0);
        let b = b.clamp(0.0, 1.0);

        let max = (size - 1) as f32;
        let r_coord = (r * max).round() as u32;
        let g_coord = (g * max).round() as u32;
        let b_coord = (b * max).round() as u32;

        let index = ((b_coord * size * size + g_coord * size + r_coord) * 4) as usize;

        (
            data[index],
            data[index + 1],
            data[index + 2],
            data[index + 3],
        )
    }

    fn expected_quantized(value: f32, size: u32) -> f32 {
        let max = (size - 1) as f32;
        ((value.clamp(0.0, 1.0) * max).round()) / max
    }

    #[test]
    fn test_lut_corners() {
        let size = 16;
        let data = generate_identity_lut_3d(size);

        assert_eq!(sample_lut_3d(&data, size, 0.0, 0.0, 0.0), (0, 0, 0, 255));
        assert_eq!(sample_lut_3d(&data, size, 1.0, 0.0, 0.0), (255, 0, 0, 255));
        assert_eq!(sample_lut_3d(&data, size, 0.0, 1.0, 0.0), (0, 255, 0, 255));
        assert_eq!(sample_lut_3d(&data, size, 0.0, 0.0, 1.0), (0, 0, 255, 255));
        assert_eq!(
            sample_lut_3d(&data, size, 1.0, 1.0, 1.0),
            (255, 255, 255, 255)
        );
    }

    #[test]
    fn test_lut_identity_mapping() {
        let size = 16;
        let data = generate_identity_lut_3d(size);

        let test_values = [0.0, 0.25, 0.5, 0.75, 1.0];

        for &r in &test_values {
            for &g in &test_values {
                for &b in &test_values {
                    let (out_r, out_g, out_b, out_a) = sample_lut_3d(&data, size, r, g, b);

                    let out_r_norm = out_r as f32 / 255.0;
                    let out_g_norm = out_g as f32 / 255.0;
                    let out_b_norm = out_b as f32 / 255.0;

                    let er = expected_quantized(r, size);
                    let eg = expected_quantized(g, size);
                    let eb = expected_quantized(b, size);

                    assert!((out_r_norm - er).abs() < 0.01);
                    assert!((out_g_norm - eg).abs() < 0.01);
                    assert!((out_b_norm - eb).abs() < 0.01);
                    assert_eq!(out_a, 255);
                }
            }
        }
    }

    #[test]
    fn test_lut_size() {
        let size = 8;
        let data = generate_identity_lut_3d(size);

        assert_eq!(data.len(), (size * size * size * 4) as usize);
    }

    #[test]
    fn test_lut_grayscale_diagonal() {
        let size = 32;
        let data = generate_identity_lut_3d(size);

        for i in 0..=10 {
            let value = i as f32 / 10.0;
            let (r, g, b, a) = sample_lut_3d(&data, size, value, value, value);

            let q = expected_quantized(value, size);
            let expected = (q * 255.0).round() as u8;

            assert_eq!(r, expected);
            assert_eq!(g, expected);
            assert_eq!(b, expected);
            assert_eq!(a, 255);
        }
    }

    #[test]
    fn test_lut_primary_colors() {
        let size = 16;
        let data = generate_identity_lut_3d(size);

        let q = expected_quantized(0.5, size);
        let expected = (q * 255.0).round() as u8;

        assert_eq!(
            sample_lut_3d(&data, size, 0.5, 0.0, 0.0),
            (expected, 0, 0, 255)
        );
        assert_eq!(
            sample_lut_3d(&data, size, 0.0, 0.5, 0.0),
            (0, expected, 0, 255)
        );
        assert_eq!(
            sample_lut_3d(&data, size, 0.0, 0.0, 0.5),
            (0, 0, expected, 255)
        );
    }

    #[test]
    fn test_lut_alpha_always_opaque() {
        let size = 8;
        let data = generate_identity_lut_3d(size);

        for i in (0..data.len()).step_by(4) {
            assert_eq!(data[i + 3], 255);
        }
    }
}
