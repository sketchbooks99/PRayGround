#include "shader.h"

namespace oprt
{

// --------------------------------------------------------------------
Shader::Shader()
{

}

Shader::Shader(const std::filesystem::path& vert_name, const std::filesystem::path& frag_name, const std::filesystem::path& geom_name)
{

}

// --------------------------------------------------------------------
void Shader::begin() const 
{

}

void Shader::end() const
{

}

// --------------------------------------------------------------------
void Shader::addSource(const std::string& source, GLuint type)
{
    
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