#pragma once

#include "util.h"
#include "shape.h"
#include "texture.h"

namespace oprt {

/**
 * @brief Area light. Emittance is evaluated at the closest-hit program associated with a geometry.
 */
struct AreaEmitterData {
    
};

class AreaEmitter {
public:
    AreaEmitter(Shape* shape, Texture* texture) {}


private:
    Shape* m_shape;
    Texture* m_texture;
    float strength;
};

/**
 * @brief Environment emitter. In general, emittance is evaluated at a miss program.
 * 
 * @note 
 * - If you'd like to render with image based lighting, you must be use latitude-longitude format (sphere map).
 * - EnvironmentEmitter allows ordinary textures such as checker or constant.
 */
struct EnvironmentEmitterData {

};

class EnvironmentEmitter {
public:
    explicit EnvironmentEmitter(const std::string& filename);
    explicit EnvironmentEmitter(Texture* texture);
private:
    Texture* m_texture;
};

}
