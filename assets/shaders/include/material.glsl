#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

struct GpuMaterial {
    vec4 base_color;
    float metallic;
    float roughness;
    uint base_color_tex;
    uint normal_tex;
    uint metallic_roughness_tex;
    uint ao_tex;
    vec2 _padding;
    uint flags;
};

const uint HAS_BASE_COLOR        = 1u << 0;
const uint HAS_NORMAL            = 1u << 1;
const uint HAS_METALROUGH       = 1u << 2;
const uint HAS_AO                = 1u << 3;

bool has_base_color_texture(GpuMaterial m) {
    return (m.flags & HAS_BASE_COLOR) != 0u;
}

bool has_normal_texture(GpuMaterial m) {
    return (m.flags & HAS_NORMAL) != 0u;
}

bool has_metalrough_texture(GpuMaterial m) {
    return (m.flags & HAS_METALROUGH) != 0u;
}

bool has_ao_texture(GpuMaterial m) {
    return (m.flags & HAS_AO) != 0u;
}

bool uses_any_textures(GpuMaterial m) {
    return m.flags != 0u;
}

bool uses_all_textures(GpuMaterial m) {
    return (m.flags & (HAS_BASE_COLOR | HAS_NORMAL | HAS_METALROUGH | HAS_AO)) ==
           (HAS_BASE_COLOR | HAS_NORMAL | HAS_METALROUGH | HAS_AO);
}

#endif
