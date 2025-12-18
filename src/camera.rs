use crate::input_state::InputState;
use crate::math::perspective;
use nalgebra::{Matrix4, Point3, Vector3};

pub struct Camera {
    pub position: Vector3<f32>,
    pub forward: Vector3<f32>,
    pub right: Vector3<f32>,
    pub up: Vector3<f32>,

    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub fov_degrees: f32,
}

impl Camera {
    pub fn new() -> Self {
        let mut cam = Self {
            position: Vector3::new(0.0, 2.0, 5.0),
            forward: Vector3::z_axis().into_inner(),
            right: Vector3::x_axis().into_inner(),
            up: Vector3::y_axis().into_inner(),

            yaw: -90.0_f32.to_radians(),
            pitch: 0.0,
            speed: 5.0, // m/s
            fov_degrees: 70.0,
        };
        cam.update_basis();
        cam
    }

    fn update_basis(&mut self) {
        // Standard FPS camera equation
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();

        self.forward = Vector3::new(cy * cp, sp, sy * cp).normalize();
        self.right = self.forward.cross(&Vector3::y()).normalize();
        self.up = self.right.cross(&self.forward).normalize();
    }

    pub fn update_from_input(&mut self, input: &InputState, dt: f32) {
        // Movement
        let mut move_dir = Vector3::zeros();
        if input.forward {
            move_dir += self.forward;
        }
        if input.backward {
            move_dir -= self.forward;
        }
        if input.left {
            move_dir -= self.right;
        }
        if input.right {
            move_dir += self.right;
        }
        if input.up {
            move_dir += self.up;
        }
        if input.down {
            move_dir -= self.up;
        }

        if move_dir != Vector3::zeros() {
            self.position += move_dir.normalize() * self.speed * dt;
        }

        // Rotation (only while RMB is pressed)
        if input.rotating {
            let (dx, dy) = input.mouse_delta;
            const SENS: f32 = 0.002;
            self.yaw += dx * SENS;
            self.pitch -= dy * SENS;
            self.pitch = self.pitch.clamp(-1.55, 1.55);

            self.update_basis();
        }
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(
            &Point3::from(self.position),
            &Point3::from(self.position + self.forward),
            &self.up,
        )
    }

    pub fn projection_matrix(&self, aspect: f32) -> Matrix4<f32> {
        perspective(self.fov_degrees, aspect, 0.1, None, None)
    }

    pub fn sun_direction_view_space(&self) -> [f32; 4] {
        let az = 0.8_f32;
        let ph = 0.6_f32;

        let sun_world =
            Vector3::new(ph.cos() * az.sin(), ph.sin(), ph.cos() * az.cos()).normalize();

        let view = self.view_matrix();
        let view_rot = view.fixed_view::<3, 3>(0, 0);
        let sun_view = (view_rot * sun_world).normalize();

        [sun_view.x, sun_view.y, sun_view.z, 0.0]
    }
}
