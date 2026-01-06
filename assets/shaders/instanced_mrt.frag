#version 460
#include "preamble.glsl"
#include "buffers.glsl"

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uvs;
layout(location = 2) flat in uint v_material_id;

layout(location = 0) out vec4 normal_mrt;
layout(location = 1) out vec4 uvs_mrt;
layout(location = 2) out vec4 albedo_mrt;
layout(location = 3) out uint material_id;

layout(set = 1, binding = 0) uniform sampler2D textures[];
layout(set = 1, binding = 1) uniform sampler samplers[];

layout(push_constant) uniform PC {
    Transforms transforms;
    MaterialIds material_ids;
    Materials mesh_materials;
};


void main() {
    GpuMaterial mat = mesh_materials.materials[v_material_id];

    uint tex_index = mat.base_color_tex;

    vec4 albedo = texture(
    textures[nonuniformEXT(tex_index)],
    v_uvs
    );

    normal_mrt = vec4(normalize(v_normal), 0.0);
    uvs_mrt = vec4(v_uvs, 0.0, 0.0);
    albedo_mrt = albedo * mat.base_color;
    material_id = v_material_id;
}
