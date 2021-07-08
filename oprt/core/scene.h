#pragma once

#include "primitive.h"
#include "bitmap.h"
#include "../optix/context.h"
#include "../emitter/envmap.h"
#include <sutil/Camera.h>
// #include "../oprt.h"

namespace oprt {

class Scene {
public:
    Scene() {}

    void createOnDevice();
    void freeFromDevice();

    void render();

    /** 
     * @brief Create programs associated with primitives.
     */
    void createHitgroupPrograms(const Context& ctx, const Module& module);

    /**
     * @brief Return all hitgroup programs contained in Scene
     */
    std::vector<ProgramGroup> hitgroupPrograms();

    /** 
     * @brief Create SBT with HitGroupData. 
     * @note SBTs for raygen and miss program aren't created at here.
     */
    void createHitgroupSBT(OptixShaderBindingTable& sbt);

    void addPrimitiveInstance(PrimitiveInstance ps) {
        ps.sort();
        if (m_primitive_instances.empty())
            ps.setSbtIndexBase(0);
        else
            ps.setSbtIndexBase(m_primitive_instances.back().sbtIndex());
        m_primitive_instances.push_back(ps); 
    }
    std::vector<PrimitiveInstance> primitiveInstances() const { return m_primitive_instances; }

    void setEnvironment(const float3& color) 
    { 
        m_environment = std::make_shared<EnvironmentEmitter>(color);
    }
    void setEnvironment(const std::shared_ptr<Texture>& texture)
    {
        m_environment = std::make_shared<EnvironmentEmitter>(texture);
    }
    void setEnvironment(const std::shared_ptr<EnvironmentEmitter>& env) { m_environment = env; }
    void setEnvironment(const std::filesystem::path& filename)
    {
        m_environment = std::make_shared<EnvironmentEmitter>(filename);
    }
    std::shared_ptr<EnvironmentEmitter> environment() const { return m_environment; }

    void setDepth(unsigned int d) { m_depth = d; }
    unsigned int depth() const { return m_depth; }

    void setSamplesPerLaunch(unsigned int samples_per_launch) { m_samples_per_launch = samples_per_launch; }
    unsigned int samplesPerLaunch() const { return m_samples_per_launch; }

    void setNumSamples(unsigned int num_samples) { m_num_samples = num_samples; }
    unsigned int numSamples() const { return m_num_samples; }

    void setCamera(const sutil::Camera& camera) { m_camera = camera; }
    const sutil::Camera& camera() const { return m_camera; }

    void setFilm(const std::shared_ptr<Bitmap>& film) { m_film = film; }
    std::shared_ptr<Bitmap> film() const { return m_film; }
private:
    std::vector<PrimitiveInstance> m_primitive_instances;   // Primitive instances with same transformation.
    std::shared_ptr<EnvironmentEmitter> m_environment;      // Environment map
    unsigned int m_depth;                                   // Maximum depth of ray tracing.
    unsigned int m_samples_per_launch;                      // Specify the number of samples per call of optixLaunch.
    unsigned int m_num_samples;                             // The number of samples per pixel for non-interactive mode.
    sutil::Camera m_camera;                                 // Camera
    /// @note For future work, This should be a vector of bitmap to enable AOV.
    std::shared_ptr<Bitmap> m_film;                         // Film of rendering
};

}