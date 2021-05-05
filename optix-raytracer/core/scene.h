#pragma once

#include "../core/primitive.h"
#include <sutil/Camera.h>
#include "../optix-raytracer.h"

namespace oprt {

class Scene {
public:
    Scene() {}

    /** 
     * @brief Create programs associated with primitives.
     */
    void create_hitgroup_programs(const OptixDeviceContext& ctx, const Module& module);

    /**
     * @brief Return all hitgroup programs contained in Scene
     */
    std::vector<ProgramGroup> hitgroup_programs();

    /** 
     * @brief Create SBT with HitGroupData. 
     * @note SBTs for raygen and miss program aren't created at here.
     */
    void create_hitgroup_sbt(OptixShaderBindingTable& sbt);

    void add_primitive_instance(const PrimitiveInstance& ps) { m_primitive_instances.push_back(ps); }
    std::vector<PrimitiveInstance> primitive_instances() const { return m_primitive_instances; }
    std::vector<PrimitiveInstance>& primitive_instances() { return m_primitive_instances; }

    void set_width(const unsigned int w) { m_width = w; }
    unsigned int width() const { return m_width; }

    void set_height(const unsigned int h) { m_height = h; }
    unsigned int height() const { return m_height; }

    void set_bgcolor(const float4& bg) { m_bgcolor = bg; }
    float4 bgcolor() const { return m_bgcolor; }

    void set_depth(unsigned int d) { m_depth = d; }
    unsigned int depth() const { return m_depth; }

    void set_samples_per_launch(unsigned int samples_per_launch) { m_samples_per_launch = samples_per_launch; }
    unsigned int samples_per_launch() const { return m_samples_per_launch; }

    void set_num_samples(unsigned int num_samples) { m_num_samples = num_samples; }
    unsigned int num_samples() const { return m_num_samples; }

    void set_camera(const sutil::Camera& camera) { m_camera = camera; }
    sutil::Camera camera() const { return m_camera; }
private:
    std::vector<PrimitiveInstance> m_primitive_instances;   // Primitive instances with same transformation.
    unsigned int m_width, m_height;                         // Dimensions of output result.
    float4 m_bgcolor;                                       // Background color
    unsigned int m_depth;                                   // Maximum depth of ray tracing.
    unsigned int m_samples_per_launch;                      // Specify the number of samples per call of optixLaunch.
    unsigned int m_num_samples;                             // The number of samples per pixel for non-interactive mode.
    sutil::Camera m_camera;                                 // Camera
};

}