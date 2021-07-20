#include "shader.h"

namespace gl
{

namespace fs = std::filesystem;

// --------------------------------------------------------------------
Shader::Shader()
{

}

Shader::Shader(const fs::path& vert_name, const fs::path& frag_name, const fs::path& geom_name)
{
    load(vert_name, frag_name, geom_name);
}

// --------------------------------------------------------------------
void Shader::begin() const 
{

}

void Shader::end() const
{

}

// --------------------------------------------------------------------
void Shader::load(const fs::path& vert_name, const fs::path& frag_name, const fs::path& geom_name)
{
    m_program = glCreateProgram();

    {
        auto vert_path = findDataPath(vert_name);
        Assert(vert_path, "The shader file '" + vert_name.string() + "' is not found.");

        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
    }

    {
        auto frag_path = findDataPath(frag_name);
        Assert(frag_path, "The shader file '" + frag_name.string() + "' is not found.");

        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
    }

    if (geom_name != "")
    {
        auto geom_path = findDataPath(geom_name);
        Assert(geom_path, "The shader file '" + geom_name.string() + "' is not found.");
    }
}

// --------------------------------------------------------------------
void Shader::addSource(const std::string& source, GLuint type)
{
    m_sources.emplace(type, source);
}

// --------------------------------------------------------------------
void Shader::create()
{

}

GLuint Shader::program() const 
{ 
    return m_program; 
}

// --------------------------------------------------------------------
void Shader::setUniform1f(const std::string& name, float v1) const
{
    
}
void Shader::setUniform2f(const std::string& name, float v1, float v2) const
{
    
}
void Shader::setUniform3f(const std::string& name, float v1, float v2, float v3) const
{
    
}
void Shader::setUniform4f(const std::string& name, float v1, float v2, float v3, float v4) const
{
    
}

// --------------------------------------------------------------------
void Shader::setUniform2f(const std::string& name, const float2& vec) const
{
    
}
void Shader::setUniform3f(const std::string& name, const float3& vec) const
{
    
}
void Shader::setUniform4f(const std::string& name, const float4& vec) const
{
    
}

// --------------------------------------------------------------------
void Shader::setUniform1i(const std::string& name, int32_t v1) const
{
    
}
void Shader::setUniform2i(const std::string& name, int32_t v1, int32_t v2) const
{
    
}
void Shader::setUniform3i(const std::string& name, int32_t v1, int32_t v2, int32_t v3) const
{
    
}
void Shader::setUniform4i(const std::string& name, int32_t v1, int32_t v2, int32_t v3, int32_t v4) const
{
    
}

// --------------------------------------------------------------------
void Shader::setUniform1fv(const std::string& name, const float* v) const
{

}
void Shader::setUniform2fv(const std::string& name, const float* v) const
{

}
void Shader::setUniform3fv(const std::string& name, const float* v) const
{
    
}
void Shader::setUniform4fv(const std::string& name, const float* v) const
{
    
}

// --------------------------------------------------------------------
void Shader::setUniform1iv(const std::string& name, const int32_t* v) const
{

}
void Shader::setUniform2iv(const std::string& name, const int32_t* v) const
{

}
void Shader::setUniform3iv(const std::string& name, const int32_t* v) const
{
    
}
void Shader::setUniform4iv(const std::string& name, const int32_t* v) const
{
    
}

}