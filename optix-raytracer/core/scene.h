#pragma once

#include "../core/primitive.h"
#include <sutil/Camera.h>
#include "../core/optix-raytracer.h"

namespace oprt {

class Scene {
public:
    Scene() {}

    /** 
     * \brief Create programs associated with primitives.
     */
    void create_hitgroup_programs(const OptixDeviceContext& ctx, const Module& module) {
        for (auto &ps : m_primitive_instances) { 
            for (auto &p : ps.primitives()) {
                p.create_programs(ctx, (OptixModule)module);
            }
        }
    }

    std::vector<ProgramGroup> hitgroup_programs() {
        std::vector<ProgramGroup> program_groups;
        for (auto &ps : m_primitive_instances) {
            for (auto &p : ps.primitives()) {
                std::vector<ProgramGroup> programs = p.program_groups();
                std::copy(programs.begin(), programs.end(), std::back_inserter(program_groups));
            }
        }
        return program_groups;
    }

    /** 
     * \brief Create SBT with HitGroupData. 
     * \note SBTs for raygen and miss program aren't created at here.
     */
    void create_hitgroup_sbt(OptixShaderBindingTable& sbt) {
        size_t hitgroup_record_size = sizeof(HitGroupRecord);
        unsigned int sbt_idx = 0;
        std::vector<HitGroupRecord> hitgroup_records;
        for (auto &ps : m_primitive_instances) {
            for (auto &p : ps.primitives()) {
                for (int i=0; i<RAY_TYPE_COUNT; i++) {
                    // Bind sbt to radiance program groups. 
                    if (i == 0) 
                    {
                        hitgroup_records.push_back(HitGroupRecord());
                        p.bind_radiance_record(&hitgroup_records.back());
                        hitgroup_records.back().data.shapedata = reinterpret_cast<void*>(p.shape()->get_dptr());
                        hitgroup_records.back().data.matdata = reinterpret_cast<void*>(p.material()->get_dptr());
                        hitgroup_records.back().data.sample_func_idx = (unsigned int)p.materialtype();
                    } 
                    // Bind sbt to occlusion program groups.
                    else if (i == 1) 
                    {
                        hitgroup_records.push_back(HitGroupRecord());
                        p.bind_occlusion_record(&hitgroup_records.back());
                        hitgroup_records.back().data.shapedata = reinterpret_cast<void*>(p.shape()->get_dptr());
                    }
                }
            }
        }

        CUDABuffer<HitGroupRecord> d_hitgroup_records;
        d_hitgroup_records.alloc_copy(hitgroup_records);

        sbt.hitgroupRecordBase = d_hitgroup_records.d_ptr();
        sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
        sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_records.size());
    }

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
private:
    std::vector<PrimitiveInstance> m_primitive_instances;   // Primitive instances with same transformation.
    unsigned int m_width, m_height;                         // Dimensions of output result.
    float4 m_bgcolor;                                       // Background color
    unsigned int m_depth;                                   // Maximum depth of ray tracing.
    unsigned int m_samples_per_launch;                      // Specify the number of samples per call of optixLaunch.
    unsigned int m_num_samples;
};

}