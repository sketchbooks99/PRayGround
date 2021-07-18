#pragma once 

#include "../core/util.h"
#include <glad/glad.h>

namespace oprt
{

class Shader 
{
public: 
    Shader();
    Shader(const std::filesystem::path& vert_name, const std::filesystem::path& frag_name, const std::filesystem::path& geom_name = "");

    void begin() const;
    void end() const;

    /**
     * @brief
     * Add the source with the specific type of shader.
     * @note
     * If the shader of the same type as the t specified in the argument has been already existed, the shader with such type will be overwritten.
     */
    void addSource( const std::string& source, GLuint type );

    void create();
    GLuint program() const { return m_program; }

private:
    std::unordered_map<GLuint, std::string> m_sources;
    GLuint m_program;
};

class ComputeShader 
{
public:
    ComputeShader(const std::filesystem::path& filename);
private:
    std::string m_source;
};

}