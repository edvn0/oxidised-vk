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
            // Finite reverse-Z projection
            // Maps: n -> 1.0, far -> 0.0
            // 
            // For reverse-Z, the depth mapping is:
            // clip.z / clip.w = (a * view.z + b) / (-view.z)
            // At near (view.z = -n): should give 1.0
            // At far  (view.z = -far): should give 0.0
            //
            // Solving:
            // (a * (-n) + b) / n = 1  =>  -a*n + b = n  =>  b = n + a*n = n(1+a)
            // (a * (-far) + b) / far = 0  =>  -a*far + b = 0  =>  b = a*far
            //
            // So: n(1+a) = a*far
            //     n + a*n = a*far
            //     n = a*far - a*n = a(far - n)
            //     a = n / (far - n)
            //     b = a*far = (n*far) / (far - n)
            
            let a = n / (far - n);
            let b = (n * far) / (far - n);
            
            // Column-major order: each group of 4 is a COLUMN, not a row
            let p = Matrix4::from_column_slice(&[
                x,   0.0,  0.0,  0.0,  // Column 0
                0.0, y,    0.0,  0.0,  // Column 1
                0.0, 0.0,  a,   -1.0,  // Column 2 (z mapping)
                0.0, 0.0,  b,    0.0,  // Column 3 (w mapping)
            ]);
            
            if let Some(inv) = inverse {
                *inv = Matrix4::from_column_slice(&[
                    1.0 / x, 0.0,       0.0,        0.0,
                    0.0,     1.0 / y,   0.0,        0.0,
                    0.0,     0.0,       0.0,       -1.0,
                    0.0,     0.0,       1.0 / b,    a / b,
                ]);
            }
            p
        }
        None => {
            // Infinite reverse-Z projection (recommended)
            // Maps: n -> 1.0, infinity -> 0.0
            let a = 0.0;    // For infinite projection
            let b = n;      // Near plane distance
            
            // Column-major order
            let p = Matrix4::from_column_slice(&[
                x,   0.0,  0.0,  0.0,  // Column 0
                0.0, y,    0.0,  0.0,  // Column 1
                0.0, 0.0,  a,   -1.0,  // Column 2
                0.0, 0.0,  b,    0.0,  // Column 3
            ]);
            
            if let Some(inv) = inverse {
                *inv = Matrix4::from_column_slice(&[
                    1.0 / x, 0.0,       0.0,        0.0,
                    0.0,     1.0 / y,   0.0,        0.0,
                    0.0,     0.0,       0.0,       -1.0,
                    0.0,     0.0,      -1.0 / b,    0.0,
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
        
        println!("\n=== Infinite Reverse-Z Projection ===");
        println!("Matrix (column-major):");
        for row in 0..4 {
            println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", 
                proj[(row, 0)], proj[(row, 1)], proj[(row, 2)], proj[(row, 3)]);
        }
        
        // Column 2 should be [0, 0, 0, -1] for infinite reverse-Z
        assert_eq!(proj[(0, 2)], 0.0, "m[0,2] should be 0");
        assert_eq!(proj[(1, 2)], 0.0, "m[1,2] should be 0");
        assert_eq!(proj[(2, 2)], 0.0, "m[2,2] should be 0 for infinite");
        assert_eq!(proj[(3, 2)], -1.0, "m[3,2] should be -1");
        
        // Column 3 should be [0, 0, near, 0]
        assert_eq!(proj[(0, 3)], 0.0, "m[0,3] should be 0");
        assert_eq!(proj[(1, 3)], 0.0, "m[1,3] should be 0");
        assert!(proj[(2, 3)] > 0.0, "m[2,3] should be near plane distance");
        assert_eq!(proj[(3, 3)], 0.0, "m[3,3] should be 0");
    }
    
    #[test]
    fn test_reverse_z_finite_projection() {
        let proj = perspective(90.0, 1.0, 0.1, Some(100.0), None);
        
        println!("\n=== Finite Reverse-Z Projection ===");
        println!("Matrix (column-major):");
        for row in 0..4 {
            println!("  [{:8.4}, {:8.4}, {:8.4}, {:8.4}]", 
                proj[(row, 0)], proj[(row, 1)], proj[(row, 2)], proj[(row, 3)]);
        }
        
        // Test that near plane (z=0.1) maps to clip.z/clip.w = 1.0
        let near_point = nalgebra::Vector4::new(0.0, 0.0, -0.1, 1.0);  // -z in view space
        let clip = proj * near_point;
        let ndc_z = clip.z / clip.w;
        println!("Near point NDC z: {:.6} (should be ~1.0)", ndc_z);
        assert!((ndc_z - 1.0).abs() < 0.01, "Near plane should map to 1.0");
        
        // Test that far plane maps to ~0.0
        let far_point = nalgebra::Vector4::new(0.0, 0.0, -100.0, 1.0);
        let clip = proj * far_point;
        let ndc_z = clip.z / clip.w;
        println!("Far point NDC z: {:.6} (should be ~0.0)", ndc_z);
        assert!(ndc_z < 0.1, "Far plane should map close to 0.0");
    }
}