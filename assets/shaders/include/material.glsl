#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

struct GpuMaterial {
    vec4 base_color;

    uint base_color_tex;
    uint normal_tex;
    uint metallic_roughness_tex;
    uint ao_tex;
    uint emissive_tex;
    uint flags;

    float metallic;
    float roughness;
    float ao;
    float emissive;
    float alpha_cutoff;

    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
};

const uint MATERIAL_FLAG_DOUBLE_SIDED = 1 << 0;  // Disable backface culling
const uint MATERIAL_FLAG_ALPHA_BLEND = 1 << 1;   // Use alpha blending
const uint MATERIAL_FLAG_ALPHA_TEST = 1 << 2;    // Use alpha testing (cutoff)
const uint MATERIAL_FLAG_UNLIT = 1 << 3;         // Skip lighting (emissive only)
const uint MATERIAL_FLAG_CAST_SHADOWS = 1 << 4;  // Material casts shadows

#endif
