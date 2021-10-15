#pragma once 

#include <glad/glad.h>
#include <prayground/core/util.h>
#include <prayground/math/matrix.h>
#include <unordered_map>
#include <filesystem>

namespace prayground {

namespace gl {

/**
 * @brief 
 * Wrapper class for OpenGL shader
 */

inline std::string getGLErrorTypeString(GLenum err)
{
    switch (err)
    {
        case GL_NO_ERROR:           return "No error";
        case GL_INVALID_ENUM:       return "Invalid enum";
        case GL_INVALID_VALUE:      return "Invalid value";
        case GL_INVALID_OPERATION:  return "Invalid operation";
        case GL_STACK_OVERFLOW:     return "Stack overflow";
        case GL_STACK_UNDERFLOW:    return "Stack underflow";
        case GL_OUT_OF_MEMORY:      return "Out of memory";
        //case GL_TABLE_TOO_LARGE:    return "Table too large";
        default:                    return "Unkwown GL error";
    }
}

inline std::string getGLShaderTypeString(uint32_t type)
{
    switch (type)
    {
        case GL_VERTEX_SHADER:      return "Vertex shader";
        case GL_FRAGMENT_SHADER:    return "Fragment shader";
        case GL_GEOMETRY_SHADER:    return "Geometry shader";
        case GL_COMPUTE_SHADER:     return "Compute shader";
        default:                    return "Unknown shader type";
    }
}

class Shader 
{
public: 
    Shader();
    Shader(const std::filesystem::path& vert_name, const std::filesystem::path& frag_name);
    Shader(const std::filesystem::path& vert_name, const std::filesystem::path& frag_name, const std::filesystem::path& geom_name);

    /** Specify the range to apply the shader in a C++ code. */ 
    void begin() const;
    void end() const;

    /** Load shader sources from files. */
    void load(const std::filesystem::path& vert_name, const std::filesystem::path& frag_name);
    void load(const std::filesystem::path& vert_name, const std::filesystem::path& frag_name, const std::filesystem::path& geom_name);

    /**
     * @brief
     * Add the source with the specific type of shader.
     * @note
     * If the shader of the same type as the one specified in the argument already exists, 
     * the shader with that type will be overwritten.
     */
    void addSource(const std::string& source, uint32_t type);

    void create();
    uint32_t program() const;

    void bindDefaultAttributes();

    /** Setter of uniform variables */
    void setUniform1f(const std::string& name, float v1) const;
    void setUniform2f(const std::string& name, float v1, float v2) const;
    void setUniform3f(const std::string& name, float v1, float v2, float v3) const;
    void setUniform4f(const std::string& name, float v1, float v2, float v3, float v4) const;

    void setUniform2f(const std::string& name, const float2& vec) const;
    void setUniform3f(const std::string& name, const float3& vec) const;
    void setUniform4f(const std::string& name, const float4& vec) const;

    void setUniform1i(const std::string& name, int32_t v1) const;
    void setUniform2i(const std::string& name, int32_t v1, int32_t v2) const;
    void setUniform3i(const std::string& name, int32_t v1, int32_t v2, int32_t v3) const;
    void setUniform4i(const std::string& name, int32_t v1, int32_t v2, int32_t v3, int32_t v4) const;

    void setUniform1fv(const std::string& name, const float* v) const;
    void setUniform2fv(const std::string& name, const float* v) const;
    void setUniform3fv(const std::string& name, const float* v) const;
    void setUniform4fv(const std::string& name, const float* v) const;

    void setUniform1iv(const std::string& name, const int32_t* v) const;
    void setUniform2iv(const std::string& name, const int32_t* v) const;
    void setUniform3iv(const std::string& name, const int32_t* v) const;
    void setUniform4iv(const std::string& name, const int32_t* v) const;

    void setUniformMatrix2fv(const std::string& name, const Matrix2f& m) const;
    void setUniformMatrix3fv(const std::string& name, const Matrix3f& m) const;
    void setUniformMatrix4fv(const std::string& name, const Matrix4f& m) const;

protected:
    static uint32_t _createGLShaderFromSource( const std::string& source, uint32_t type );
    static uint32_t _createGLShaderFromFile( const std::filesystem::path& relative_path, uint32_t type );

private:
    std::unordered_map<uint32_t, std::string> m_sources;
    uint32_t m_program { 0 };
};

// #ifdef glDispatchCompute

class ComputeShader : public Shader
{
public:
    ComputeShader();
    ComputeShader(const std::filesystem::path& filename);

    /** Specify the range to apply the shader in a C++ code. */ 
    void begin();
    void end();

    /** Dispatch a kernel with the specified block size. */
    void dispatchCompute(uint32_t x, uint32_t y, uint32_t z) const;

    /**
     * @brief
     * Add the source with the specific type of shader.
     * @note
     * If the shader of the same type as the one specified in the argument already exists, 
     * the shader with that type will be overwritten.
     */
    void attachSource();
    void create();
private:
    std::string m_source;
    uint32_t m_program;
};

// #endif

} // ::gl

} // ::prayground