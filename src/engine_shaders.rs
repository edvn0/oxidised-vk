pub(crate) mod fullscreen_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450
            layout(location = 0) out vec2 uvs;
            void main() {
                uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                gl_Position = vec4(uvs * 2.0 - 1.0, 0.0, 1.0);
            }
        ",
    }
}

pub(crate) mod predepth {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            vulkan_version: "1.3",
            spirv_version: "1.6",
            path: "assets/shaders/predepth.vert",
            include: ["./assets/shaders/include"],
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450
                void main() { }
            "
        }
    }
}

pub(crate) mod mrt {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            vulkan_version: "1.3",
            spirv_version: "1.6",
            path: "assets/shaders/instanced_mrt.vert",
            include: ["./assets/shaders/include"],
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            vulkan_version: "1.3",
            spirv_version: "1.6",
            path: "./assets/shaders/instanced_mrt.frag",
            include: ["./assets/shaders/include"],
        }
    }
}

pub(crate) mod fullscreen {
    vulkano_shaders::shader! {
        ty: "fragment",
        vulkan_version: "1.3",
        spirv_version: "1.6",
        path: "./assets/shaders/fullscreen.frag",
        include: ["./assets/shaders/include"],
    }
}

pub(crate) mod composite {
    vulkano_shaders::shader! {
        ty: "fragment",
        vulkan_version: "1.3",
        spirv_version: "1.6",
        path: "./assets/shaders/composite.frag",
        include: ["./assets/shaders/include"],
    }
}

pub(crate) mod mrt_light {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "assets/shaders/mrt_lighting.frag",
        include: ["./assets/shaders/include"],
        vulkan_version: "1.3",
        spirv_version: "1.6",
    }
}
