#pragma once

#include "../core/primitive.h"
#include <sutil/Camera.h>
#include "../optix-raytracer.h"

namespace oprt {

class Scene {
public:
    Scene() {}

    void createSceneOnDevice();

    void render();

    /** 
     * @brief Create programs associated with primitives.
     */
    void createHitgroupPrograms(const OptixDeviceContext& ctx, const Module& module);

    /**
     * @brief Return all hitgroup programs contained in Scene
     */
    std::vector<ProgramGroup> hitgroupPrograms();

    /** 
     * @brief Create SBT with HitGroupData. 
     * @note SBTs for raygen and miss program aren't created at here.
     */
    void createHitgroupSBT(OptixShaderBindingTable& sbt);

    void addPrimitiveInstance(const PrimitiveInstance& ps) { m_primitive_instances.push_back(ps); }
    std::vector<PrimitiveInstance> primitiveInstances() const { return m_primitive_instances; }

    void setWidth(unsigned int w) { m_width = w; }
    unsigned int width() const { return m_width; }

    void setHeight(unsigned int h) { m_height = h; }
    unsigned int height() const { return m_height; }

    void setResolution(unsigned int w, unsigned int h) { m_width = w; m_height = h; }
    uint2 resolution() const { return make_uint2(m_width, m_height); }

    void setEnvironment(const float4& env) { m_environment = env; }
    float4 environment() const { return m_environment; }

    void setDepth(unsigned int d) { m_depth = d; }
    unsigned int depth() const { return m_depth; }

    void setSamplesPerLaunch(unsigned int samples_per_launch) { m_samples_per_launch = samples_per_launch; }
    unsigned int samplesPerLaunch() const { return m_samples_per_launch; }

    void setNumSamples(unsigned int num_samples) { m_num_samples = num_samples; }
    unsigned int numSamples() const { return m_num_samples; }

    void setCamera(const sutil::Camera& camera) { m_camera = camera; }
    const sutil::Camera& camera() const { return m_camera; }
private:
    std::vector<PrimitiveInstance> m_primitive_instances;   // Primitive instances with same transformation.
    unsigned int m_width, m_height;                         // Dimensions of output result.
    float4 m_environment;                                       // Background color
    unsigned int m_depth;                                   // Maximum depth of ray tracing.
    unsigned int m_samples_per_launch;                      // Specify the number of samples per call of optixLaunch.
    unsigned int m_num_samples;                             // The number of samples per pixel for non-interactive mode.
    sutil::Camera m_camera;                                 // Camera
    Bitmap<uchar4> m_film;
};

}