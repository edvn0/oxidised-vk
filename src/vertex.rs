use meshopt_rs::vertex::Position;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex, Clone, Copy, Default)]
#[repr(C)]
pub struct StandardMeshVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    uvs: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    tangent: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    bitangent: [f32; 3],
}

#[derive(BufferContents, Vertex, Clone, Copy, Default)]
#[repr(C)]
pub struct PositionMeshVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

impl StandardMeshVertex {
    pub fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uvs: [f32; 2],
        tangent: [f32; 3],
        bitangent: [f32; 3],
    ) -> Self {
        Self {
            position,
            uvs,
            normal,
            tangent,
            bitangent,
        }
    }
}

impl Position for StandardMeshVertex {
    fn pos(&self) -> [f32; 3] {
        self.position
    }
}

impl PositionMeshVertex {
    pub fn new(position: [f32; 3]) -> Self {
        Self { position }
    }
}

impl Position for PositionMeshVertex {
    fn pos(&self) -> [f32; 3] {
        self.position
    }
}
