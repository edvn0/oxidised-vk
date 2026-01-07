#version 460
#include "preamble.glsl"
#include "buffers.glsl"

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uvs;
layout(location = 2) in vec3 v_tangent;
layout(location = 3) in vec3 v_bitangent;
layout(location = 4) flat in uint v_material_id;

layout(location = 0) out vec4 albedo_mrt;
layout(location = 1) out vec4 normal_mrt;
layout(location = 2) out vec4 material_mrt; // Emissive, Metallic, Roughness, AO

layout(set = 1, binding = 0) uniform sampler2D textures[];
layout(set = 1, binding = 1) uniform sampler samplers[];

layout(push_constant) uniform PC {
    Transforms transforms;
    MaterialIds material_ids;
    Materials mesh_materials;
};

// ============================================================================
// Helper: Sample texture or return default value
// ============================================================================
vec4 sample_texture_or_default(uint tex_index, vec2 uv, vec4 default_value) {
    if (tex_index == 0 ) {
        return default_value;
    }
    return texture(textures[nonuniformEXT(tex_index)], uv);
}

float sample_texture_or_default_r(uint tex_index, vec2 uv, float default_value) {
    if (tex_index == 0 ) {
        return default_value;
    }
    return texture(textures[nonuniformEXT(tex_index)], uv).r;
}
// ============================================================================
// Helper: Convert sRGB to linear (if textures are in sRGB space)
// ============================================================================
vec3 srgb_to_linear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

vec3 linear_to_srgb(vec3 linear) {
    return pow(linear, vec3(1.0 / 2.2));
}
// ============================================================================
// Main geometry pass
// ============================================================================

void main() {
    GpuMaterial mat = mesh_materials.materials[v_material_id];

    // ------------------------------------------------------------------------
    // 1. Sample base color (albedo)
    // ------------------------------------------------------------------------
    vec4 albedo_sample = sample_texture_or_default(
        mat.base_color_tex,
        v_uvs,
        vec4(1.0) // Default white if no texture
    );

    // Apply base color factor (tint)
    vec3 albedo = albedo_sample.rgb * mat.base_color.rgb;
    float alpha = albedo_sample.a * mat.base_color.a;

    // ------------------------------------------------------------------------
    // 2. Sample and process normal
    // ------------------------------------------------------------------------
    vec3 N;

    // Check if material has a normal map
    if (mat.normal_tex != 0) {
        // Sample normal map (tangent space)
        vec3 tangent_normal = texture(
            textures[nonuniformEXT(mat.normal_tex)],
            v_uvs
        ).xyz;

        tangent_normal = tangent_normal * 2.0 - 1.0;

        // TODO: If you have tangent/bitangent, transform to world space:
        mat3 TBN = mat3(
            normalize(v_tangent),
            normalize(v_bitangent),
            normalize(v_normal)
        );

        vec3 N = normalize(TBN * tangent_normal);
    } else {
        N = normalize(v_normal);
    }

    // ------------------------------------------------------------------------
    // 3. Sample metallic and roughness
    // ------------------------------------------------------------------------
    float metallic;
    float roughness;

    // Check if material has metallic-roughness texture
    if (mat.metallic_roughness_tex != 0 ) {
        // glTF 2.0 standard packing:
        // R = Occlusion (optional, we'll use separate AO texture)
        // G = Roughness
        // B = Metallic
        vec3 mr_sample = texture(
            textures[nonuniformEXT(mat.metallic_roughness_tex)],
            v_uvs
        ).rgb;

        roughness = mr_sample.g * mat.roughness;
        metallic = mr_sample.b * mat.metallic;
    } else {
        // Use material constants if no texture
        metallic = mat.metallic;
        roughness = mat.roughness;
    }

    // Clamp to valid ranges
    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.04, 1.0);

    // ------------------------------------------------------------------------
    // 4. Sample ambient occlusion
    // ------------------------------------------------------------------------
    float ao;

    if (mat.ao_tex != 0) {
        // AO is typically stored in red channel
        ao = texture(
            textures[nonuniformEXT(mat.ao_tex)],
            v_uvs
        ).r * mat.ao;
    } else {
        ao = mat.ao;
    }

    ao = clamp(ao, 0.0, 1.0);

    // ------------------------------------------------------------------------
    // 5. Sample emissive
    // ------------------------------------------------------------------------
    float emissive;

    if (mat.emissive_tex != 0) {
        // Emissive is often RGB, but we'll pack just intensity for now
        vec3 emissive_color = texture(
            textures[nonuniformEXT(mat.emissive_tex)],
            v_uvs
        ).rgb;

        // Convert to intensity (luminance)
        emissive = dot(emissive_color, vec3(0.2126, 0.7152, 0.0722));
        emissive *= mat.emissive;
    } else {
        emissive = mat.emissive;
    }

    emissive = clamp(emissive, 0.0, 1.0);

    // ------------------------------------------------------------------------
    // 6. Write to G-Buffer outputs
    // ------------------------------------------------------------------------

    // Albedo (already in linear space)
    albedo_mrt = vec4(albedo, alpha);

    // Normal (in world space or view space, depending on your setup)
    normal_mrt = vec4(N, 0.0);

    // Material properties packed into RGBA
    material_mrt = vec4(
        emissive,   // R = emissive intensity [0,1]
        metallic,   // G = metallic [0,1]
        roughness,  // B = roughness [0.04,1]
        ao          // A = ambient occlusion [0,1]
    );
}
