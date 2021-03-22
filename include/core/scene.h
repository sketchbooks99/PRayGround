#pragma once

#include <include/core/primitive.h>
#include <sutil/Camera.h>
#include <include/core/pathtracer.h>

namespace pt {

class Scene {
public:
    Scene() {}

    /** 
     * \brief Create programs associated with primitives.
     */
    void create_hitgroup_programs(const OptixDeviceContext& ctx, const OptixModule& module) {
        for (auto &ps : m_primitive_instances) { 
            for (auto &p : ps.primitives()) {
                p.create_programs(ctx, module);
            }
        }
    }

    std::vector<ProgramGroup> hitgroup_programs() {
        std::vector<ProgramGroup> program_groups;
        for (auto &ps : m_primitive_instances) {
            for (auto &p : ps.primitives()) {
                std::vector<ProgramGroup> programs = p.program_groups();
                /**
                 * \brief Insert hitgroup programs of primitive at end of vector.
                 * 
                 * \note \c ProgramGroup is implicitly casted to \c OptixProgramGroup at here.
                 */
                std::copy(programs.begin(), programs.end(), std::back_inserter(program_groups));
            }
        }
        return program_groups;
    }

    /** 
     * \brief Create SBT with HitGroupData. 
     * \note SBTs for raygen and miss program aren't created at here.
     */
    void create_hitgroup_sbt(const OptixModule& module, OptixShaderBindingTable& sbt) {
        size_t hitgroup_record_size = sizeof(HitGroupRecord);
        unsigned int sbt_idx = 0;
        unsigned int num_hitgroup_records = 0;
        std::vector<HitGroupRecord> hitgroup_records;
        for (auto &ps : m_primitive_instances) {
            for (auto &p : ps.primitives()) {
                for (int i=0; i<RAY_TYPE_COUNT; i++) {
                    // Bind sbt to radiance program groups. 
                    if (i == 0) 
                    {
                        HitGroupRecord record;
                        record.data.shapedata = p.shape()->get_dptr();
                        record.data.matptr = p.material()->get_dptr();
                        p.bind_radiance_record(record);
                        hitgroup_records.push_back(record);
                        sbt_idx++;
                    } 
                    // Bind sbt to occlusion program groups.
                    else if (i == 1) 
                    {
                        HitGroupRecord record;
                        memset(&record, 0, hitgroup_record_size);
                        p.bind_occlusion_record(record);
                        hitgroup_records.push_back(record);
                        sbt_idx++;
                    }
                    num_hitgroup_records++;
                }
            }
        }

        // Prepare the device side data buffer of HitGroupRecord.
        CUdeviceptr d_hitgroup_records;
        CUDA_CHECK(cudaMalloc( 
            reinterpret_cast<void**>(&d_hitgroup_records), 
            hitgroup_record_size * num_hitgroup_records 
            ));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_hitgroup_records), 
            hitgroup_records.data(), 
            hitgroup_record_size * num_hitgroup_records, 
            cudaMemcpyHostToDevice 
            ));

        sbt.hitgroupRecordBase = d_hitgroup_records;
        sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroup_record_size );
        sbt.hitgroupRecordCount = num_hitgroup_records;
    }

    void add_primitive_instance(const PrimitiveInstance& ps) { m_primitive_instances.push_back(ps); }
    std::vector<PrimitiveInstance> primitive_instances() const { return m_primitive_instances; }
    std::vector<PrimitiveInstance>& primitive_instance() { return m_primitive_instances; }

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
    std::vector<PrimitiveInstance> m_primitive_instances;  // Primitive instances with same transformation.
    unsigned int m_width, m_height;                         // Dimensions of output result.
    float4 m_bgcolor;                                       // Background color
    unsigned int m_depth;                                   // Maximum depth of ray tracing.
    unsigned int m_samples_per_launch;                      // Specify the number of samples per call of optixLaunch.
};

}