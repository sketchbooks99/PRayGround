#include "shader.h"
#include <prayground/core/file_util.h>

namespace prayground {

namespace gl
{

namespace fs = std::filesystem;

// --------------------------------------------------------------------
Shader::Shader()
{

}

Shader::Shader(const fs::path& vert_name, const fs::path& frag_name)
{
    load(vert_name, frag_name);
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
void Shader::load(const fs::path& vert_name, const fs::path& frag_name)
{
    m_program = glCreateProgram();

    GLuint vert_shader = 0, frag_shader = 0;

    // Vertex shader
    {
        std::optional<fs::path> vert_path = pgFindDataPath(vert_name);
        ASSERT(vert_path, "The shader file '" + vert_name.string() + "' is not found.");

        vert_shader = _createGLShaderFromFile(vert_path.value(), GL_VERTEX_SHADER);
        glAttachShader(m_program, vert_shader);
    }

    // Fragment shader
    {
        std::optional<fs::path> frag_path = pgFindDataPath(frag_name);
        ASSERT(frag_path, "The shader file '" + frag_name.string() + "' is not found.");

        frag_shader = _createGLShaderFromFile(frag_path.value(), GL_FRAGMENT_SHADER);
        glAttachShader(m_program, frag_shader);
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
        pgLogFatal("Linking of program failed:", infolog);

        return;
    }

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
}

void Shader::load(const fs::path& vert_name, const fs::path& frag_name, const fs::path& geom_name)
{
    m_program = glCreateProgram();

    GLuint vert_shader = 0, frag_shader = 0, geom_shader = 0;

    // Vertex shader
    {
        std::optional<fs::path> vert_path = pgFindDataPath(vert_name);
        ASSERT(vert_path, "The shader file '" + vert_name.string() + "' is not found.");

        vert_shader = _createGLShaderFromFile(vert_path.value(), GL_VERTEX_SHADER);
        glAttachShader(m_program, vert_shader);
    }

    // Fragment shader
    {
        std::optional<fs::path> frag_path = pgFindDataPath(frag_name);
        ASSERT(frag_path, "The shader file '" + frag_name.string() + "' is not found.");

        frag_shader = _createGLShaderFromFile(frag_path.value(), GL_FRAGMENT_SHADER);
        glAttachShader(m_program, frag_shader);
    }

    // Geometry shader
    {
        std::optional<fs::path> geom_path = pgFindDataPath(geom_name);
        ASSERT(geom_path, "The shader file '" + geom_name.string() + "' is not found.");

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
        pgLogFatal("Shader::load(): Linking of program failed:", infolog);

        return;
    }
    
    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);
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
    UNIMPLEMENTED();
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
        pgLogFatal("The shader program hasn't been created yet.");
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
    glUniform2f(glGetUniformLocation(m_program, name.c_str()), v1, v2);
}
void Shader::setUniform3f(const std::string& name, float v1, float v2, float v3) const
{
    glUniform3f(glGetUniformLocation(m_program, name.c_str()), v1, v2, v3);   
}
void Shader::setUniform4f(const std::string& name, float v1, float v2, float v3, float v4) const
{   
    glUniform4f(glGetUniformLocation(m_program, name.c_str()), v1, v2, v3, v4);
}

// --------------------------------------------------------------------
void Shader::setUniform2f(const std::string& name, const float2& vec) const
{   
    glUniform2f(glGetUniformLocation(m_program, name.c_str()), vec.x, vec.y);
}
void Shader::setUniform3f(const std::string& name, const float3& vec) const
{
    glUniform3f(glGetUniformLocation(m_program, name.c_str()), vec.x, vec.y, vec.z);
}
void Shader::setUniform4f(const std::string& name, const float4& vec) const
{
    glUniform4f(glGetUniformLocation(m_program, name.c_str()), vec.x, vec.y, vec.z, vec.w);
}

// --------------------------------------------------------------------
void Shader::setUniform1i(const std::string& name, int32_t v1) const
{
    glUniform1i(glGetUniformLocation(m_program, name.c_str()), v1);
}
void Shader::setUniform2i(const std::string& name, int32_t v1, int32_t v2) const
{
    glUniform2i(glGetUniformLocation(m_program, name.c_str()), v1, v2);
}
void Shader::setUniform3i(const std::string& name, int32_t v1, int32_t v2, int32_t v3) const
{   
    glUniform3i(glGetUniformLocation(m_program, name.c_str()), v1, v2, v3);
}
void Shader::setUniform4i(const std::string& name, int32_t v1, int32_t v2, int32_t v3, int32_t v4) const
{   
    glUniform4i(glGetUniformLocation(m_program, name.c_str()), v1, v2, v3, v4);
}

// --------------------------------------------------------------------
void Shader::setUniform1fv(const std::string& name, int32_t n, const float* v) const
{
    glUniform1fv(glGetUniformLocation(m_program, name.c_str()), n, v);
}
void Shader::setUniform2fv(const std::string& name, int32_t n, const float2* v) const
{
    const float* data = reinterpret_cast<const float*>(v);
    glUniform2fv(glGetUniformLocation(m_program, name.c_str()), n, data);
}
void Shader::setUniform3fv(const std::string& name, int32_t n, const float3* v) const
{
    const float* data = reinterpret_cast<const float*>(v);
    glUniform3fv(glGetUniformLocation(m_program, name.c_str()), n, data);
}
void Shader::setUniform4fv(const std::string& name, int32_t n, const float4* v) const
{   
    const float* data = reinterpret_cast<const float*>(v);
    glUniform4fv(glGetUniformLocation(m_program, name.c_str()), n, data);
}

// --------------------------------------------------------------------
void Shader::setUniform1iv(const std::string& name, int32_t n, const int32_t* v) const
{
    glUniform1iv(glGetUniformLocation(m_program, name.c_str()), n, v);
}
void Shader::setUniform2iv(const std::string& name, int32_t n, const int2* v) const
{
    const int32_t* data = reinterpret_cast<const int32_t*>(v);
    glUniform2iv(glGetUniformLocation(m_program, name.c_str()), n, data);
}
void Shader::setUniform3iv(const std::string& name, int32_t n, const int3* v) const
{
    const int32_t* data = reinterpret_cast<const int32_t*>(v);
    glUniform3iv(glGetUniformLocation(m_program, name.c_str()), n, data);
}
void Shader::setUniform4iv(const std::string& name, int32_t n, const int4* v) const
{
    const int32_t* data = reinterpret_cast<const int32_t*>(v);
    glUniform4iv(glGetUniformLocation(m_program, name.c_str()), n, data);
}

// --------------------------------------------------------------------
void Shader::setUniformMatrix2f(const std::string& name, const Matrix2f& m) const
{
    const float* data = m.data();
    glUniformMatrix2fv(glGetUniformLocation(m_program, name.c_str()), 1, /* transpose = */ false, data);
}

void Shader::setUniformMatrix3f(const std::string& name, const Matrix3f& m) const
{
    const float* data = m.data();
    glUniformMatrix3fv(glGetUniformLocation(m_program, name.c_str()), 1, /* transpose = */ false, data);
}

void Shader::setUniformMatrix4f(const std::string& name, const Matrix4f& m) const
{
    const float* data = m.data();
    glUniformMatrix4fv(glGetUniformLocation(m_program, name.c_str()), 1, /* transpose = */ false, data);
}

// --------------------------------------------------------------------
void Shader::setUniformMatrix2fv(const std::string& name, int32_t n, const Matrix2f* m) const 
{
    // Concat all matrices data to single ptr.
    float* data = new float[4 * n];
    for (int i = 0; i < n; i++)
        memcpy(&data[i * 4], m[i].data(), sizeof(float) * 4);
    glUniformMatrix2fv(glGetUniformLocation(m_program, name.c_str()), n, /* transpose = */ false, data);
}

void Shader::setUniformMatrix3fv(const std::string& name, int32_t n, const Matrix3f* m) const 
{
    // Concat all matrices data to single ptr.
    float* data = new float[9 * n];
    for (int i = 0; i < n; i++)
        memcpy(&data[i * 9], m[i].data(), sizeof(float) * 9);
    glUniformMatrix2fv(glGetUniformLocation(m_program, name.c_str()), n, /* transpose = */ false, data);
}

void Shader::setUniformMatrix4fv(const std::string& name, int32_t n, const Matrix4f* m) const 
{
    // Concat all matrices data to single ptr.
    float* data = new float[16 * n];
    for (int i = 0; i < n; i++)
        memcpy(&data[i * 16], m[i].data(), sizeof(float) * 16);
    glUniformMatrix2fv(glGetUniformLocation(m_program, name.c_str()), n, /* transpose = */ false, data);
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
    ASSERT(err == GL_NO_ERROR, "OpenGL generated error: " + toString(getGLErrorTypeString(err)) + ".");

    if (compiled == GL_FALSE) 
    {
        // Get buffer size to store the compilation log
        GLint bufsize;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &bufsize);
        
        GLsizei length;
        GLchar* infolog = (GLchar*)malloc(bufsize);
        glGetShaderInfoLog(shader, bufsize, &length, infolog);

        glDeleteShader(shader);
        THROW("Compilation of " + getGLShaderTypeString(type) + "failed:" + infolog);
    }

    return shader;
}

GLuint Shader::_createGLShaderFromFile(const fs::path& relative_path, GLuint type)
{
    std::optional<fs::path> filepath = pgFindDataPath(relative_path);
    ASSERT(filepath, "The shader file '" + relative_path.string() + "' is not found.");

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
    catch (const std::istream::failure& e)
    {
        THROW("Failed to load shader source from file due to '" + std::string(e.what()) + "'.");
    }

    return _createGLShaderFromSource(source, type);
}

} // ::gl

} // ::prayground