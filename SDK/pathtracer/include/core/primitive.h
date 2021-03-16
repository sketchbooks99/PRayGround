#pragma once

#include <core/util.h>
#include <core/shape.h>
#include <core/material.h>
#include <core/transform.h>
#include <optix/program.h>

namespace pt {

class Primitive {
public:
    Primitive(ShapePtr shape_ptr, MaterialPtr material_ptr, const Transform& transform, uint32_t sbt_index)
    : m_shape_ptr(m_shape_ptr), m_material_ptr(m_material_ptr), m_transform(transform), m_sbt_index(sbt_index) {
        m_program_groups.resize(RAY_TYPE_COUNT);
        for (auto &pg : m_program_groups) {
            pg = ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
        }
    }

    // Create programs based on shape type. 
    void create_programs(const OptixDeviceContext& ctx, const OptixModule& module) {
        Assert(!m_program_groups.empty(), "ProgramGroup is not allocated.");
        if (shapetype() == ShapeType::Mesh) {
            m_program_groups[0].create(ctx, ProgramEntry(module, ch_func_str(shape_map[shapetype()]) ) );
            if (m_program_groups.size() > 1)
                m_program_groups[1].create(ctx, ProgramEntry(module, CH_FUNC_STR("occlusion") ) );
        } else {
            m_program_groups[0].create(ctx, ProgramEntry(module, ch_func_str(shape_map[shapetype()]) ), 
                                            ProgramEntry(module, is_func_str(shape_map[shapetype()]) ) );
            if (m_program_groups.size() > 1)
                m_program_groups[1].create(ctx, ProgramEntry(module, CH_FUNC_STR("occlusion") ) );
        }
    }

    // Preparing (alloc and copy) shape data to the device. 
    void prepare_shapedata() { m_shape_ptr->prepare_data(); }

    // Configure the OptixBuildInput from shape data.
    void build_input(OptixBuildInput& bi) { m_shape_ptr->build_input( bi, m_sbt_index ); }

    /** 
     * \brief 
     * Free temporal device side pointers. 
     * \note  
     * Currently, only aabb_buffer is freed
     */
    void free_temp_buffer() { if (m_shape_ptr->type() != ShapeType::Mesh) m_shape_ptr->free_aabb_buffer(); }

    // Bind programs and HitGroupRecord
    void bind_radiance_record(HitGroupRecord record) {
        Assert(!m_program_groups.empty(), "ProgramGroups is not allocated.");
        m_program_groups[0].bind_sbt_and_program(record);
    }
    void bind_occlusion_record(HitGroupRecord record) {
        Assert(m_program_groups.size() > 1, "Occlusion program is not contained in rendering.");
        m_program_groups[1].bind_sbt_and_program(record);
    }

    // Getter 
    uint32_t sbt_index() const  { return m_sbt_index; }
    MaterialPtr material() const { return m_material_ptr; }
    ShapePtr shape() const  { return m_shape_ptr; }
    Transform transform() const { return m_transform; }
    ShapeType shapetype() const { return m_shape_ptr->type(); }
    MaterialType materialtype() const { return m_material_ptr->type(); }

    std::vector<ProgramGroup> program_groups() { return m_program_groups; }

private:
    // Member variables.
    ShapePtr m_shape_ptr;
    MaterialPtr m_material_ptr;
    Transform m_transform;

    /** 
     * @param
     * - 0 : for radiance program
     * - 1 : for occlusion program (optional)
     **/
    std::vector<ProgramGroup> m_program_groups;
    // For managing sbt index which associated with a shader.
    uint32_t m_sbt_index { 0 };
};

}