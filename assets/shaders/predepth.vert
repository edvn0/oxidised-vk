#version 460

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0, std140) uniform UBO {
    mat4 view;
    mat4 projection;
    mat4 inverse_projection;
    vec4 sun_direction;
};

struct Transform {
    mat4 t;
};
layout(set = 1, binding = 1, std140) readonly buffer SSBO {
    Transform transforms[];
};

void main() {
    mat4 model = transforms[gl_InstanceIndex] . t;
    vec4 world_pos = model * vec4(position, 1.0);
    gl_Position = projection * ( view * world_pos );
}
