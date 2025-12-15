pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        vulkan_version: "1.3",
        spirv_version: "1.6",
        src: r#"
#version 460

#extension GL_EXT_buffer_reference : require

layout(location=0) out vec4 out_color;
layout(location=1) out vec2 out_uv;

struct Vertex {
    float x; float y;
    float u; float v;
    uint rgba;
};

layout(buffer_reference, std430) readonly buffer VB {
    Vertex v[];
};

layout(push_constant) uniform PC {
    vec4 lrtb;
    VB vb;
    uint texture_id;
    uint sampler_id;
} pc;

void main() {
    Vertex vert = pc.vb.v[gl_VertexIndex];
    out_color = unpackUnorm4x8(vert.rgba);
    out_uv = vec2(vert.u, vert.v);

    float L=pc.lrtb.x, R=pc.lrtb.y;
    float T=pc.lrtb.z, B=pc.lrtb.w;

    mat4 proj = mat4(
        2/(R-L),0,0,0,
        0,2/(T-B),0,0,
        0,0,-1,0,
        (R+L)/(L-R),(T+B)/(B-T),0,1
    );

    gl_Position = proj * vec4(vert.x, vert.y, 0, 1);
}
"# }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r#"
#version 460
layout(location=0) in vec4 in_color;
layout(location=1) in vec2 in_uv;
layout(location=0) out vec4 out_color;

void main() {
    out_color = in_color;
}
"# }
}
