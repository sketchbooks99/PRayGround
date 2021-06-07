#include "../core/emitter.h"

/**
 * @brief Environment emitter. In general, emittance is evaluated at a miss program.
 * 
 * @note 
 * - If you'd like to render with image based lighting, you must be use latitude-longitude format (sphere map).
 * - EnvironmentEmitter allows ordinary textures such as checker or constant.
 */

namespace oprt {

struct EnvironmentEmitterData {

};

class EnvironmentEmitter {
public:
    explicit EnvironmentEmitter(const std::string& filename);
    explicit EnvironmentEmitter(const float3& c)
    : m_texture(c) {}
    explicit EnvironmentEmitter(const std::shared_ptr<Texture>& texture)
    : m_texture(texture) {}
private:
    std::shared_ptr<Texture> m_texture;
};

}