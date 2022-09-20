#pragma once

#include <glad/glad.h>
#include <iostream>

namespace prayground {
    namespace gl {

        inline std::string typeToString(GLuint gltype)
        {
            switch (gltype)
            {
            case GL_FLOAT:
                return "GL_FLOAT";
            case GL_UNSIGNED_BYTE:
                return "GL_UNSIGNED_BYTE";
            case GL_UNSIGNED_INT:
                return "GL_UNSIGNED_INT";
            case GL_RED:
                return "GL_RED";
            case GL_R8:
                return "GL_R8";
            case GL_R32F:
                return "GL_R32F";
            case GL_RG:
                return "GL_RG";
            case GL_RG8:
                return "GL_RG8";
            case GL_RG32F:
                return "GL_RG32F";
            case GL_RGB:
                return "GL_RGB";
            case GL_RGB8:
                return "GL_RGB8";
            case GL_RGB32F:
                return "GL_RGB32F";
            case GL_RGBA:
                return "GL_RGBA";
            case GL_RGBA8:
                return "GL_RGBA8";
            case GL_RGBA32F:
                return "GL_RGBA32F";
            case GL_LINEAR:
                return "GL_LINEAR";
            case GL_NEAREST:
                return "GL_NEAREST";
            case GL_LINEAR_MIPMAP_LINEAR:
                return "GL_LINEAR_MIPMAP_LINEAR";
            case GL_LINEAR_MIPMAP_NEAREST:
                return "GL_LINEAR_MIPMAP_NEAREST";
            case GL_NEAREST_MIPMAP_LINEAR:
                return "GL_NEAREST_MIPMAP_LINEAR";
            case GL_NEAREST_MIPMAP_NEAREST:
                return "GL_NEAREST_MIPMAP_NEAREST";
            case GL_CLAMP_TO_BORDER:
                return "GL_CLAMP_TO_BORDER";
            case GL_CLAMP_TO_EDGE:
                return "GL_CLAMP_TO_EDGE";
            }
        }
    } // namespace gl

} // namespace prayground