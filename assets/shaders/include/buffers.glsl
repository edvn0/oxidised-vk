#ifndef BUFFERS_GLSL
#define BUFFERS_GLSL


struct Transform {
    mat4 model;
};

layout(buffer_reference, std430) readonly buffer Transforms {
    Transform ts[];
};

layout(buffer_reference, std430) readonly buffer MaterialIds {
    uint mi[];
};

#include "material.glsl"
layout(buffer_reference, std430) readonly buffer Materials {
    GpuMaterial materials[];
};

#endif
