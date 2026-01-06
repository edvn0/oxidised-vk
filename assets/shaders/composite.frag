#version 460
#include "preamble.glsl"

 layout(location = 0) in vec2 v_uvs;
 layout(location = 0) out vec4 out_color;

 layout(set=0, binding=0) uniform sampler2D hdr_tex;
 layout(set=0, binding=1) uniform sampler2D bloom_tex;

 layout(push_constant) uniform PC {
     float exposure;
     float bloom_strength;
 } pc;

 void main() {
     vec3 hdr   = texture(hdr_tex, v_uvs).rgb;
     vec3 bloom = texture(bloom_tex, v_uvs).rgb;

     vec3 color = hdr + bloom * pc.bloom_strength;

     color *= pc.exposure;

     out_color = vec4(color, 1.0);
 }
