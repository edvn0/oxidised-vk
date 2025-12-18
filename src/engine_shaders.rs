
pub(crate) mod predepth {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            vulkan_version: "1.3",
            spirv_version: "1.6",
            path: "assets/shaders/predepth.vert",
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

    pub mod vs {
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

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                        #version 450
                        layout(location = 0) in vec2 uvs;
                        layout(location = 0) out vec4 f_color;

                        layout(set=0, binding=0) uniform sampler2D composite_output;

                        vec3 aces(vec3 x) {
                            return clamp((x*(2.51*x+0.03)) / (x*(2.43*x+0.59)+0.14), 0.0, 1.0);
                        }

                        vec3 linear_to_srgb(vec3 x){
                            return pow(x, vec3(1.0/2.2));
                        }

                        void main(){
                            vec3 color = texture(composite_output, uvs).xyz;
                            color = linear_to_srgb(aces(color));
                            f_color = vec4(color,1.0);
                        }
                    ",
        }
    }
}

pub(crate) mod composite {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                        #version 450
                        layout(location = 0) out vec2 v_uvs;
                        void main(){
                            v_uvs = vec2((gl_VertexIndex<<1)&2, gl_VertexIndex&2);
                            gl_Position = vec4(v_uvs*2.0 + -1.0,0.0,1.0);
                        }
                    "
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                        #version 450

                        layout(location = 0) in vec2 v_uvs;
                        layout(location = 0) out vec4 out_color;

                        layout(set=0, binding=0) uniform sampler2D hdr_tex;
                        layout(set=0, binding=1) uniform sampler2D bloom_tex;

                        layout (push_constant) uniform PC {
                            float bloom_strength;
                        };

                        void main() {
                            vec3 hdr = texture(hdr_tex, v_uvs).rgb;
                            vec3 bloom = texture(bloom_tex, v_uvs).rgb;

                            float bloom_final_strength = max(bloom_strength, 1.0);
                            out_color = vec4(hdr + bloom * bloom_final_strength, 1.0);
                        }
                    "
        }
    }
}

pub(crate) mod mrt_light {
    pub mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                        #version 450
                        layout(location = 0) out vec2 v_uvs;
                        void main() {
                            v_uvs = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
                            gl_Position = vec4(v_uvs * 2.0 + -1.0, 0.0, 1.0);
                        }
                    "
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "assets/shaders/mrt_lighting.frag",
        }
    }
}