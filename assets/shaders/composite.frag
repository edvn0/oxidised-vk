#version 460
#include "preamble.glsl"

layout(location = 0) in vec2 v_uvs;
layout(location = 0) out vec4 out_color;

// Textures
layout(set = 0, binding = 0) uniform sampler2D hdr_tex;
layout(set = 0, binding = 1) uniform sampler2D bloom_tex;
layout(set = 0, binding = 2) uniform sampler3D grading_lut;

// Push constants
layout(push_constant) uniform PC {
    // Tone mapping
    float exposure;
    float gamma;

    // Bloom
    float bloom_strength;

    // Color grading
    float saturation;
    float contrast;
    float brightness;

    // Vignette
    float vignette_strength;
    float vignette_radius;

    // Film grain
    float grain_strength;
    float time;

    // Output
    int tonemap_operator;
    int _pad0;
} pc;

// ------------------------------------------------------------------
// Utility
// ------------------------------------------------------------------

float luminance(vec3 c)
{
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float hash(vec2 p)
{
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p . x * p . y);
}

// ------------------------------------------------------------------
// Tone mapping
// ------------------------------------------------------------------

vec3 tonemap(vec3 c)
{
    if (pc . tonemap_operator == 0) {
        // Reinhard
        return c / ( c + 1.0 );
    } else {
        // ACES (Narkowicz approximation)
        const float a = 2.51;
        const float b = 0.03;
        const float c2 = 2.43;
        const float d = 0.59;
        const float e = 0.14;
        return clamp(( c * ( a * c + b ) ) / ( c * ( c2 * c + d ) + e ), 0.0, 1.0);
    }
}

// ------------------------------------------------------------------
// Color grading
// ------------------------------------------------------------------

vec3 apply_color_grading(vec3 c)
{
    // Saturation
    float luma = luminance(c);
    c = mix(vec3(luma), c, pc . saturation);

    // Contrast
    c = ( c - 0.5 ) * pc . contrast + 0.5;

    // Brightness
    c += pc . brightness;

    return c;
}

// ------------------------------------------------------------------
// LUT
// ------------------------------------------------------------------

vec3 apply_lut(vec3 c)
{
    c = clamp(c, 0.0, 1.0);
    return texture(grading_lut, c) . rgb;
}

// ------------------------------------------------------------------
// Vignette
// ------------------------------------------------------------------

vec3 apply_vignette(vec3 c, vec2 uv)
{
    float d = distance(uv, vec2(0.5));
    float v = smoothstep(pc . vignette_radius, 1.0, d);
    return mix(c, c * ( 1.0 - pc . vignette_strength ), v);
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------

void main()
{
    vec3 hdr = texture(hdr_tex, v_uvs) . rgb;
    vec3 bloom = texture(bloom_tex, v_uvs) . rgb;

    // Combine
    vec3 color = hdr + bloom * pc . bloom_strength;

    // Exposure
    color *= pc . exposure;

    // Tone mapping
    color = tonemap(color);

    // Color grading
    color = apply_color_grading(color);

    // LUT
    color = apply_lut(color);

    // Vignette
    color = apply_vignette(color, v_uvs);

    // Film grain (very subtle)
    float grain = hash(v_uvs * vec2(1920.0, 1080.0) + pc . time) - 0.5;
    color += grain * pc . grain_strength;

    // Gamma / output encoding
    color = pow(color, vec3(1.0 / pc . gamma));

    out_color = vec4(color, 1.0);
}
