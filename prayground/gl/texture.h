#pragma once 

#include <glad/glad.h>

namespace prayground {

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

} // ::prayground