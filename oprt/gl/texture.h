#pragma once 

#include <glad/glad.h>

namespace gl {

class Texture 
{
public:
    struct Settings
    {

    };

    Texture();

private:
    GLuint m_tex;
};

} // ::gl