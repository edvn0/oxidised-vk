use std::f32::consts::PI;
use nalgebra::Matrix4;

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

    let projection = match f {
        Some(far) => {
            // Standard finite projection
            let a = n / (far - n);
            let b = far * a;

            let p = Matrix4::from_row_slice(&[
                x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, a, b, 0.0, 0.0, -1.0, 0.0,
            ]);

            if let Some(inv) = inverse {
                *inv = Matrix4::from_row_slice(&[
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
            let a = -1.0;
            let b = -n;

            let p = Matrix4::from_row_slice(&[
                x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 0.0, a, b, 0.0, 0.0, -1.0, 0.0,
            ]);

            if let Some(inv) = inverse {
                *inv = Matrix4::from_row_slice(&[
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
                    a / b,
                ]);
            }

            p
        }
    };

    projection
}