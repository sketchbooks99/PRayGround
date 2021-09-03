#pragma once 

#include <glad/glad.h>

namespace prayground {

namespace gl {

/**
 * @brief 
 * Frame buffer object class
 */

class Fbo {
public:
    struct Settings 
    {
        
    };

    Fbo();

    void allocate(const Settings& settings);
    void allocate(int32_t width, int32_t height);

    void begin();
    void end();

private:
};

} // ::gl

} // ::prayground