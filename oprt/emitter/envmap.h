#include "../core/emitter.h"
#include "../texture/constant.h"

/**
 * @brief Environment emitter. In general, emittance is evaluated at a miss program.
 * 
 * @note 
 * - If you'd like to render with image based lighting, you must be use latitude-longitude format (sphere map).
 * - EnvironmentEmitter allows ordinary textures such as checker or constant.
 */

namespace oprt {

struct EnvironmentEmitterData {
    void* texdata;
    unsigned int tex_func_id;
};

class EnvironmentEmitter : public Emitter {
public:
    explicit EnvironmentEmitter(const std::filesystem::path& filename);
    explicit EnvironmentEmitter(const float3& c)
    : m_texture(std::make_shared<ConstantTexture>(c)) {}
    explicit EnvironmentEmitter(const std::shared_ptr<Texture>& texture)
    : m_texture(texture) {}

    void prepareData() override;

    EmitterType type() const override { return EmitterType::Envmap; }
private:
    std::shared_ptr<Texture> m_texture;
};

}