#include "shader.h"
#include "../core/file_util.h"

namespace oprt {

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
    glUseProgram(m_program);
}

void Shader::end() const
{
    glUseProgram(0);
}

// --------------------------------------------------------------------
void Shader::load(const fs::path& vert_name, const fs::path& frag_name, const fs::path& geom_name)
{
    m_program = glCreateProgram();

    GLuint vert_shader = 0, frag_shader = 0, geom_shader = 0;

    // Vertex shader
    {
        std::optional<fs::path> vert_path = findDataPath(vert_name);
        Assert(vert_path, "The shader file '" + vert_name.string() + "' is not found.");

        vert_shader = _createGLShaderFromFile(vert_path.value(), GL_VERTEX_SHADER);
        glAttachShader(m_program, vert_shader);
    }

    // Fragment shader
    {
        std::optional<fs::path> frag_path = findDataPath(frag_name);
        Assert(frag_path, "The shader file '" + frag_name.string() + "' is not found.");

        frag_shader = _createGLShaderFromFile(frag_path.value(), GL_FRAGMENT_SHADER);
        glAttachShader(m_program, frag_shader);
    }

    // Geometry shader
    if (geom_name != "")
    {
        std::optional<fs::path> geom_path = findDataPath(geom_name);
        Assert(geom_path, "The shader file '" + geom_name.string() + "' is not found.");

        geom_shader = _createGLShaderFromFile(geom_path.value(), GL_GEOMETRY_SHADER);
        glAttachShader(m_program, geom_shader);
    }

    // Linking of shaders
    glLinkProgram(m_program);
    GLint linked;
    glGetProgramiv(m_program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE)
    {
        GLint bufsize;
        glGetProgramiv(m_program, GL_INFO_LOG_LENGTH, &bufsize);

        GLsizei length;
        GLchar* infolog = (GLchar*)malloc(bufsize);
        glGetProgramInfoLog(m_program, bufsize, &length, infolog);

        glDeleteProgram(m_program);
        Message(MSG_ERROR, "Shader::load(): Linking of program failed:", infolog);

        return;
    }
    
    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
    if (geom_shader) 
        glDeleteShader(geom_shader);
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
void Shader::bindDefaultAttributes()
{
    if (m_program == 0)
    {
        Message(MSG_ERROR, "The shader program hasn't been created yet.");
        return;
    }
    else 
    {
        glBindAttribLocation(m_program, 0, "position");
        glBindAttribLocation(m_program, 1, "color");
        glBindAttribLocation(m_program, 2, "normal");
        glBindAttribLocation(m_program, 3, "texcoord");
        return;
    }
}

// --------------------------------------------------------------------
void Shader::setUniform1f(const std::string& name, float v1) const
{
    glUniform1f(glGetUniformLocation(m_program, name.c_str()), v1);
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

// --------------------------------------------------------------------
GLuint Shader::_createGLShaderFromSource(const std::string& source, GLuint type)
{
    GLuint shader = glCreateShader(type);

    const GLchar* source_data = reinterpret_cast<const GLchar*>( source.c_str() );
    glShaderSource(shader, 1, &source_data, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        Message(MSG_ERROR, "Shader::_createGLShaderFromSource(): OpenGL generated error: " + toString(getGLErrorTypeString(err)) + ".");
        return 0;
    }

    if (compiled == GL_FALSE) 
    {
        // Get buffer size to store the compilation log
        GLint bufsize;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &bufsize);
        
        GLsizei length;
        GLchar* infolog = (GLchar*)malloc(bufsize);
        glGetShaderInfoLog(shader, bufsize, &length, infolog);

        glDeleteShader(shader);
        Message(MSG_ERROR, "Shader::_createGLShaderFromSource(): Compilation of " + getGLShaderTypeString(type) + "failed:", infolog);

        return 0;
    }

    return shader;
}

GLuint Shader::_createGLShaderFromFile(const fs::path& relative_path, GLuint type)
{
    std::optional<fs::path> filepath = findDataPath(relative_path);
    Assert(filepath, "The shader file '" + relative_path.string() + "' is not found.");

    std::string source;
    std::ifstream file_stream;
    try 
    {
        file_stream.open(filepath.value());
        std::stringstream source_stream;
        source_stream << file_stream.rdbuf();
        file_stream.close();
        source = source_stream.str();
    }
    catch (std::istream::failure e)
    {
        Message(MSG_ERROR, "gl::Shader::_createGLShaderFromFile(): Failed to create shader from file.");
    }

    return _createGLShaderFromSource(source, type);
}

} // ::gl

} // ::oprt