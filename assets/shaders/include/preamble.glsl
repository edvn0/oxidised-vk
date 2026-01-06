#ifndef PREAMBLE_GLSL
#define PREAMBLE_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_ARB_separate_shader_objects : require
#ifndef SKIP_BINDLESS
#extension GL_EXT_nonuniform_qualifier : require
#endif
#extension GL_EXT_demote_to_helper_invocation : require

#pragma optimize(on)
#pragma debug(off)


#define PI 3.14159265359
#define TAU 6.28318530718
#define HALF_PI 1.57079632679
#define DEG_TO_RAD(x) ((x) * PI / 180.0)
#define RAD_TO_DEG(x) ((x) * 180.0 / PI)

#endif
