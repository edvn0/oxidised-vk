#version 460
#include "preamble.glsl"

layout(location = 0) in vec2 v_uvs;

layout(set = 0, binding = 0) uniform UBO {
    mat4 view;
    mat4 projection;
    mat4 inverse_projection;
    vec4 sun_direction;
} ubo;

layout(set=1, binding=0) uniform sampler2D normal_tex;
layout(set=1, binding=1) uniform sampler2D uvs_tex;
layout(set=1, binding=2) uniform sampler2D albedo_tex;
layout(set=1, binding=3) uniform sampler2D depth_tex;

layout(location = 0) out vec4 out_hdr;

void main() {
    float depth = texture(depth_tex, v_uvs).r;
    if(depth >= 1.0) discard;

    vec3 N = normalize(texture(normal_tex, v_uvs).xyz);
    vec3 L = normalize(ubo.sun_direction.xyz);
    float ndotl = max(dot(N, L), 0.0);

    vec3 albedo = vec3(0.8, 0.9, 0.1) * texture(albedo_tex, v_uvs).xyz;

    out_hdr = vec4(albedo * ndotl, 1.0);
}
