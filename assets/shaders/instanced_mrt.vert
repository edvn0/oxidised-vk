#version 460
#include "preamble.glsl"
#include "buffers.glsl"

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uvs;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 bitangent;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uvs;
layout(location = 2) out vec3 v_tangent;
layout(location = 3) out vec3 v_bitangent;
layout(location = 4) flat out uint v_material_id;

#include "material.glsl"

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 view;
    mat4 projection;
    mat4 inverse_projection;
    vec4 sun_direction;
};

layout(push_constant) uniform PC {
    Transforms transforms;
    MaterialIds material_ids;
    Materials mesh_materials;
};

void main() {
    uint draw_id = gl_DrawID;
    uint object_index = gl_InstanceIndex + gl_BaseInstance;

    uint material_index = material_ids.mi[draw_id];

    mat4 model = transforms.ts[object_index].model;

    v_material_id = material_index;

    mat3 normal_mat = transpose(inverse(mat3(model)));
    vec3 world_norm = normalize(normal_mat * normal);

    vec4 world_pos = model * vec4(position, 1.0);
    vec4 view_pos = view * world_pos;

    v_uvs = uvs;

    vec3 N = normalize(normal_mat * normal);
    vec3 T = normalize(normal_mat * tangent);
    vec3 B = normalize(normal_mat * bitangent);

    mat3 view_mat = mat3(view);
    v_normal    = normalize(view_mat * N);
    v_tangent   = normalize(view_mat * T);
    v_bitangent = normalize(view_mat * B);

    gl_Position = projection * view_pos;
}
