use winit::event::{ElementState, WindowEvent, MouseButton, DeviceEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window};

pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,

    pub rotating: bool,
    pub mouse_delta: (f32, f32),
    pub last_mouse_pos: Option<(f32, f32)>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            rotating: false,
            mouse_delta: (0.0, 0.0),
            last_mouse_pos: None,
        }
    }

    pub fn end_frame(&mut self) {
        self.mouse_delta = (0.0, 0.0);
        if !self.rotating {
            self.last_mouse_pos = None;
        }
    }

    pub fn process_input(&mut self, window_event: &WindowEvent) {
        match window_event {
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.forward = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.backward = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.left = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.right = pressed,
                    PhysicalKey::Code(KeyCode::KeyR) => self.up = pressed,
                    PhysicalKey::Code(KeyCode::KeyF) => self.down = pressed,
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if *button == MouseButton::Right {
                    self.rotating = *state == ElementState::Pressed;
                    if !self.rotating {
                        self.last_mouse_pos = None;
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if !self.rotating {
                    let (x, y) = (position.x as f32, position.y as f32);
                    self.last_mouse_pos = Some((x, y));
                }
            }

            WindowEvent::Focused(false) => {
                self.rotating = false;
                self.last_mouse_pos = None;
            }

            _ => {}
        }
    }

    pub fn apply_cursor_mode(&self, window: &Window) {
        if self.rotating {
            let _ = window.set_cursor_visible(false);
            let _ = window.set_cursor_grab(CursorGrabMode::Locked);
        } else {
            let _ = window.set_cursor_visible(true);
            let _ = window.set_cursor_grab(CursorGrabMode::None);
        }
    }
}


