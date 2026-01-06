#version 460
#include "preamble.glsl"

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 view;
    mat4 projection;
    mat4 inverse_projection;
    vec4 sun_direction;
};

struct Transform {
    mat4 model;
};

layout(buffer_reference) readonly buffer Transforms {
    Transform ts[];
};

layout(push_constant) uniform PC {
    Transforms transforms;
};

void main() {
mat4 model = transforms.ts[gl_InstanceIndex].model;
    vec4 world_pos = model * vec4(position, 1.0);
    gl_Position = projection * ( view * world_pos );
}
