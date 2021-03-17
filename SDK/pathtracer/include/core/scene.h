#pragma once

#include <core/primitive.h>
#include <sutil/Camera.h>
#include <core/pathtracer.h>

namespace pt {

class Scene {
public:
    Scene() {}

    void create_hitgroup_programs(const OptixDeviceContext& ctx, const OptixModule& module);
    /** \brief build geomerty acceleration structure. */
    void build_gas(const OptixDeviceContext& ctx, AccelData& accel_data);
    void build_ias(const OptixDeviceContext& ctx, 
                   const AccelData& accel_data, 
                   const Transform& transform, 
                   std::vector<OptixInstance> instances());
    /** 
     * \brief Create SBT with HitGroupData. 
     * \note SBTs for raygen and miss program is not created at this.
     */
    void create_hitgroup_sbt(const OptixModule& module, OptixShaderBindingTable& sbt);

    void add_primitive(const PrimitiveInstance& ps) { m_primitives_instances.push_back(ps); }
    std::vector<PrimitiveInstance> primitive_instances() const { return m_primitives_instances; }

    void set_width(const unsigned int w) { m_width = w; }
    unsigned int width() const { return m_width; }

    void set_height(const unsigned int h) { m_height = h; }
    unsigned int height() const { return m_height; }

    void set_bgcolor(const float4& bg) { m_bgcolor = bg; }
    float4 bgcolor() const { return m_bgcolor; }

    void set_depth(unsigned int d) { m_depth = d; }
    unsigned int depth() const { return m_depth; }

    void set_samples_per_launch(unsigned int spl) { m_samples_per_launch = spl; }
    unsigned int samples_per_launch() const { return m_samples_per_launch; }
private:
    std::vector<PrimitiveInstance> m_primitives_instances;  // Primitive instances with same transformation.
    unsigned int m_width, m_height;                         // Dimensions of output result.
    float4 m_bgcolor;                                       // Background color
    unsigned int m_depth;                                   // Maximum depth of ray tracing.
    unsigned int m_samples_per_launch;                      // Specify the number of samples per call of optixLaunch.
};

}