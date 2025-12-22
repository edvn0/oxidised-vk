use nalgebra::Matrix4;
use std::f32::consts::PI;

/// Creates a reverse-Z perspective projection matrix
///
/// # Arguments
/// * `vertical_fov` - Vertical field of view in degrees
/// * `aspect_ratio` - Aspect ratio (width / height)
/// * `n` - Near clipping plane distance
/// * `f` - Optional far clipping plane (None = infinite)
/// * `inverse` - Optional output for inverse matrix
///
/// # Returns
/// Matrix that maps view space to clip space with reverse-Z (near=1.0, far=0.0)
pub fn perspective(
    vertical_fov: f32,
    aspect_ratio: f32,
    n: f32,
    f: Option<f32>,
    inverse: Option<&mut Matrix4<f32>>,
) -> Matrix4<f32> {
    let fov_rad = vertical_fov * 2.0 * PI / 360.0;
    let focal_length = 1.0 / (fov_rad * 0.5).tan();
    let x = focal_length / aspect_ratio;
    let y = -focal_length;

    match f {
        Some(far) => {
            let a = n / (far - n);
            let b = (n * far) / (far - n);

            let p = Matrix4::from_column_slice(&[
                x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, a, -1.0, 0.0, 0.0, b, 0.0,
            ]);

            if let Some(inv) = inverse {
                *inv = Matrix4::from_column_slice(&[
                    1.0 / x,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / y,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    1.0 / b,
                    a / b,
                ]);
            }
            p
        }
        None => {
            let a = 0.0;
            let b = n;

            let p = Matrix4::from_column_slice(&[
                x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, a, -1.0, 0.0, 0.0, b, 0.0,
            ]);

            if let Some(inv) = inverse {
                *inv = Matrix4::from_column_slice(&[
                    1.0 / x,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / y,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    0.0,
                    0.0,
                    -1.0 / b,
                    0.0,
                ]);
            }
            p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_z_infinite_projection() {
        let proj = perspective(90.0, 1.0, 0.1, None, None);

        assert_eq!(proj[(0, 2)], 0.0, "m[0,2] should be 0");
        assert_eq!(proj[(1, 2)], 0.0, "m[1,2] should be 0");
        assert_eq!(proj[(2, 2)], 0.0, "m[2,2] should be 0 for infinite");
        assert_eq!(proj[(3, 2)], -1.0, "m[3,2] should be -1");

        assert_eq!(proj[(0, 3)], 0.0, "m[0,3] should be 0");
        assert_eq!(proj[(1, 3)], 0.0, "m[1,3] should be 0");
        assert!(proj[(2, 3)] > 0.0, "m[2,3] should be near plane distance");
        assert_eq!(proj[(3, 3)], 0.0, "m[3,3] should be 0");
    }

    #[test]
    fn test_reverse_z_finite_projection() {
        let proj = perspective(90.0, 1.0, 0.1, Some(100.0), None);

        let near_point = nalgebra::Vector4::new(0.0, 0.0, -0.1, 1.0); // -z in view space
        let clip = proj * near_point;
        let ndc_z = clip.z / clip.w;
        assert!((ndc_z - 1.0).abs() < 0.01, "Near plane should map to 1.0");

        let far_point = nalgebra::Vector4::new(0.0, 0.0, -100.0, 1.0);
        let clip = proj * far_point;
        let ndc_z = clip.z / clip.w;
        assert!(ndc_z < 0.1, "Far plane should map close to 0.0");
    }
}
