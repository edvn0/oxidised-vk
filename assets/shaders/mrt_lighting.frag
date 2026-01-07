#version 460

#include "preamble.glsl"

layout(location = 0) in vec2 v_uvs;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
    mat4 inverse_projection;
    vec4 sun_direction;
} ubo;

// G-Buffer inputs
layout(set=1, binding=0) uniform sampler2D albedo_tex;
layout(set=1, binding=1) uniform sampler2D normal_tex;
layout(set=1, binding=2) uniform sampler2D material_tex;
layout(set=1, binding=3) uniform sampler2D depth_tex;

layout(location = 0) out vec4 out_hdr;

// ============================================================================
// MISSING #1: View-space position reconstruction
// ============================================================================

vec3 reconstruct_view_position(vec2 uv, float depth) {
    // Convert UV [0,1] and depth to NDC [-1,1]
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);

    // Transform to view space
    vec4 view_pos = ubo.inverse_projection * ndc;

    // Perspective divide
    return view_pos.xyz / view_pos.w;
}

// ============================================================================
// MISSING #2: PBR helper functions (Cook-Torrance BRDF)
// ============================================================================


// Normal Distribution Function (GGX/Trowbridge-Reitz)
float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ndoth = max(dot(N, H), 0.0);
    float ndoth2 = ndoth * ndoth;


    float nom = a2;
    float denom = (ndoth2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265359 * denom * denom;

    return nom / max(denom, 0.0001);
}

// Geometry Function (Smith's method with Schlick-GGX)
float geometry_schlick_ggx(float ndotv, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float nom = ndotv;
    float denom = ndotv * (1.0 - k) + k;

    return nom / max(denom, 0.0001);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float ndotv = max(dot(N, V), 0.0);
    float ndotl = max(dot(N, L), 0.0);
    float ggx2 = geometry_schlick_ggx(ndotv, roughness);
    float ggx1 = geometry_schlick_ggx(ndotl, roughness);

    return ggx1 * ggx2;
}

// Fresnel-Schlick approximation
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Fresnel with roughness (for indirect lighting)
vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ============================================================================
// MISSING #3: Complete PBR direct lighting calculation
// ============================================================================

vec3 calculate_pbr_direct_light(
    vec3 V,           // View direction (normalized)
    vec3 N,           // Normal (normalized)
    vec3 L,           // Light direction (normalized)
    vec3 radiance,    // Incoming light color * intensity
    vec3 albedo,      // Base color
    float metallic,   // Metallic value [0,1]
    float roughness   // Roughness value [0,1]
) {
    vec3 H = normalize(V + L);

    // Calculate base reflectivity (F0)
    // Dielectrics: ~0.04 (4% reflectivity)
    // Metals: use albedo as F0
    vec3 f0 = vec3(0.04);
    f0 = mix(f0, albedo, metallic);

    // Cook-Torrance BRDF components
    float ndf = distribution_ggx(N, H, roughness);
    float g = geometry_smith(N, V, L, roughness);
    vec3 f = fresnel_schlick(max(dot(H, V), 0.0), f0);

    // Calculate specular component
    vec3 numerator = ndf * g * f;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;

    // Energy conservation: kS + kD = 1.0
    vec3 ks = f; // Specular contribution
    vec3 kd = vec3(1.0) - ks;
    kd *= 1.0 - metallic; // Metals have no diffuse

    // Lambertian diffuse
    vec3 diffuse = kd * albedo / 3.14159265359;

    // Calculate outgoing radiance
    float ndotl = max(dot(N, L), 0.0);
    return (diffuse + specular) * radiance * ndotl;
}

// ============================================================================
// MISSING #4: Ambient/IBL contribution (simplified)
// ============================================================================

vec3 calculate_ambient_ibl(
    vec3 N,
    vec3 V,
    vec3 albedo,
    float metallic,
    float roughness,
    float ao
) {
    // Calculate base reflectivity
    vec3 f0 = vec3(0.04);
    f0 = mix(f0, albedo, metallic);

    // Fresnel for ambient
    vec3 ks = fresnel_schlick_roughness(max(dot(N, V), 0.0), f0, roughness);
    vec3 kd = 1.0 - ks;
    kd *= 1.0 - metallic;

    // Simple ambient approximation (replace with IBL cubemap sampling)
    // Sky color gradient based on normal
    vec3 sky_color = mix(
        vec3(0.02, 0.03, 0.04),  // Horizon (darker)
        vec3(0.05, 0.08, 0.12),  // Zenith (brighter, bluer)
        N.y * 0.5 + 0.5
    );

    vec3 irradiance = sky_color;
    vec3 diffuse = irradiance * albedo;

    // Simple specular ambient (should sample pre-filtered environment map)
    vec3 specular = sky_color * 0.5; // Placeholder

    // vec3 ambient = (kd * diffuse + specular) * ao;
vec3 ambient = kd * albedo * 0.03 * ao; // neutral-ish

    return ambient;
}

// ============================================================================
// MISSING #5: Additional light types (point lights, spot lights)
// ============================================================================

struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
    float radius;
};

vec3 calculate_point_light(
    PointLight light,
    vec3 frag_pos,
    vec3 N,
    vec3 V,
    vec3 albedo,
    float metallic,
    float roughness
) {
    vec3 L = normalize(light.position - frag_pos);
    float distance = length(light.position - frag_pos);

    // Attenuation (inverse square with radius cutoff)
    float attenuation = 1.0 / (distance * distance);
    float cutoff = 1.0 - smoothstep(light.radius * 0.75, light.radius, distance);
    attenuation *= cutoff;

    vec3 radiance = light.color * light.intensity * attenuation;

    return calculate_pbr_direct_light(V, N, L, radiance, albedo, metallic, roughness);
}

// ============================================================================
// MISSING #6: Shadow sampling (if you have shadow maps)
// ============================================================================

// TODO: Add shadow map sampling here
// float sample_shadow_map(vec3 world_pos) { ... }

// ============================================================================
// Main lighting pass
// ============================================================================

void main() {
    float depth = texture(depth_tex, v_uvs).r;
    if (depth >= 1.0) discard;

    // Sample G-buffer
    vec3 N = normalize(texture(normal_tex, v_uvs).xyz);
    vec3 albedo = texture(albedo_tex, v_uvs).rgb;

    // CRITICAL: Sample material properties
    // You need to pack this in your G-buffer pass!
    vec4 material = texture(material_tex, v_uvs);
    float metallic = material.g;   // Or however you pack it
    float roughness = material.b;
    float ao = material.a;         // Ambient occlusion

    // Reconstruct view-space position
    vec3 view_pos = reconstruct_view_position(v_uvs, depth);
    vec3 V = normalize(-view_pos);

    // ========================================================================
    // Direct lighting
    // ========================================================================

    vec3 L = normalize(ubo.sun_direction.xyz);

    // Sun light properties
    const vec3 sun_color = vec3(1.0, 0.98, 0.95);
    const float sun_intensity = 3.5;
    vec3 sun_radiance = sun_color * sun_intensity;

    // Calculate sun contribution
    vec3 Lo = calculate_pbr_direct_light(
        V, N, L,
        sun_radiance,
        albedo,
        metallic,
        roughness
    );

    // TODO: Add point lights here
    // for (int i = 0; i < num_point_lights; i++) {
    //     Lo += calculate_point_light(point_lights[i], view_pos, N, V, albedo, metallic, roughness);
    // }

    // ========================================================================
    // Indirect/ambient lighting
    // ========================================================================

    vec3 ambient = calculate_ambient_ibl(N, V, albedo, metallic, roughness, ao);

    // ========================================================================
    // Final color
    // ========================================================================

    vec3 color = Lo + ambient;
    out_hdr = vec4(color, 1.0);
}

// ============================================================================
// What you still need to implement:
// ============================================================================
//
// 1. G-Buffer material texture (metallic/roughness/AO)
//    - Pack in geometry pass: vec4(unused, metallic, roughness, ao)
//
// 2. Multiple light sources
//    - Point lights array in uniform buffer
//    - Spot lights with cone attenuation
//    - Area lights (advanced)
//
// 3. Shadow mapping
//    - Directional shadow cascades for sun
//    - Point light shadow cubemaps
//
// 4. Image-Based Lighting (IBL)
//    - Pre-filtered environment map for specular
//    - Irradiance map for diffuse
//    - BRDF integration LUT
//
// 5. Screen-space effects
//    - SSAO (Screen-space ambient occlusion)
//    - SSR (Screen-space reflections)
//
// 6. Emissive materials
//    - Add emissive channel to G-buffer
//    - Add directly to final color (no lighting needed)
//
// ============================================================================
