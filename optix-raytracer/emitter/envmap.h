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