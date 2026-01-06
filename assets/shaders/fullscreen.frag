#version 460
#include "preamble.glsl"

layout(location = 0) in vec2 uvs;
layout(location = 0) out vec4 f_color;

layout(set=0, binding=0) uniform sampler2D composite_output;
layout(set=0, binding=1) uniform sampler3D grading_lut;

vec3 apply_lut(vec3 color) {
    return texture(grading_lut, color).rgb;
}

vec3 aces(vec3 x) {
    return clamp((x*(2.51*x+0.03)) / (x*(2.43*x+0.59)+0.14), 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 x){
    return pow(x, vec3(1.0/2.2));
}

void main(){
    vec3 color = texture(composite_output, uvs).xyz;
    color = aces(color);
    color = apply_lut(color);
    color = linear_to_srgb(color);
    f_color = vec4(color,1.0);
}
