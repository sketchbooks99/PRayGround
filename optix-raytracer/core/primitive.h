/** 
 * \file primitive.h
 * \brief Management of primitives include shape and material.
 * \author Shunji Kiuchi
 * 
 * \details 
 * Primitive has shape, material and its program on CUDA 
 * to describe primitive behaivior during rendering. 
 * PrimitiveInstance has the array of primitives and transform
 * to easily construct Instance AS. 
 */
#pragma once

#include "../core/util.h"
#include "../core/shape.h"
#include "../core/material.h"
#include "../core/transform.h"
#include "../core/cudabuffer.h"
#include "../optix/program.h"
#include "../optix/sbt.h"
#include "../optix/module.h"
#include <algorithm>

namespace oprt {

class Primitive {
public:
    Primitive(Shape* shape, Material* material)
    : m_shape(shape), m_material(material) 
    {
        _initProgramGroups();
    }

    Primitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<Material>& material)
    : m_shape(shape), m_material(material) 
    {
        _initProgramGroups();
    }

    Primitive(Shape* shape, Material* material, uint32_t sbt_index)
    : m_shape(shape), m_material(material), m_sbt_index(sbt_index) 
    {
        _initProgramGroups();
    }

    Primitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<Material>& material, uint32_t sbt_index)
    : m_shape(shape), m_material(material), m_sbt_index(sbt_index) 
    {
        _initProgramGroups();
    }

    // Create programs based on shape type. 
    void createPrograms(const OptixDeviceContext& ctx, const OptixModule& module) {
        Assert(!m_program_groups.empty(), "ProgramGroup is not allocated.");
        if (shapeType() == ShapeType::Mesh) {
            // Program for mesh is only a closest-hit program. 
            m_program_groups[0].createHitgroupProgram( ctx, ProgramEntry( module, ch_func_str( shape_map[shapeType()]).c_str() ) );
            if (m_program_groups.size() > 1) {
                m_program_groups[1].createHitgroupProgram( ctx, ProgramEntry( module, ch_func_str( shape_occlusion_map[shapeType()]).c_str() ) );
            }
        } else {
            // Programs for custom primitives must include closeset-hit and intersection programs.
            m_program_groups[0].createHitgroupProgram( ctx, ProgramEntry( module, ch_func_str( shape_map[shapeType()]).c_str() ), 
                                             ProgramEntry( module, is_func_str( shape_map[shapeType()]).c_str() ) );
            if (m_program_groups.size() > 1) {
                m_program_groups[1].createHitgroupProgram( ctx, ProgramEntry( module, ch_func_str( shape_occlusion_map[shapeType()]).c_str() ),
                                                 ProgramEntry( module, is_func_str( shape_map[shapeType()]).c_str() ) );
                
            }
        
        }
    }

    // Preparing (alloc and copy) shape data to the device. 
    void prepareShapeData() { m_shape->prepareData(); }
    void prepareMaterialData() { m_material->prepareData(); }

    // Configure the OptixBuildInput from shape data.
    void buildInput( OptixBuildInput& bi ) { m_shape->buildInput( bi, m_sbt_index ); }

    /** 
     * \brief 
     * Free temporal device side pointers. 
     * \note  
     * Currently, only aabb_buffer is freed
     */
    void freeTempBuffer() { if (m_shape->type() != ShapeType::Mesh) m_shape->freeAabbBuffer(); }

    // Bind programs and HitGroupRecord
    template <typename SBTRecord>
    void bindRadianceRecord(SBTRecord* record) {
        Assert(!m_program_groups.empty(), "ProgramGroups is not allocated.");
        m_program_groups[0].bindRecord(record);
    }
    template <typename SBTRecord>
    void bindOcclusionRecord(SBTRecord* record) {
        Assert(m_program_groups.size() > 1, "Occlusion program is not contained in rendering.");
        m_program_groups[1].bindRecord(record);
    }

    // Setter
    void setSbtIndex(const uint32_t idx) { m_sbt_index = idx; } 

    // Getter 
    uint32_t sbtIndex() const { return m_sbt_index; }
    std::shared_ptr<Material> material() const { return m_material; }
    std::shared_ptr<Shape> shape() const { return m_shape; }
    ShapeType shapeType() const { return m_shape->type(); }
    MaterialType materialType() const { return m_material->type(); }

    std::vector<ProgramGroup> programGroups() const { return m_program_groups; }

private:
    void _initProgramGroups() {
        m_program_groups.resize(RAY_TYPE_COUNT);
        for (auto &pg : m_program_groups) {
            pg = ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
        }
    }

    // Member variables.
    std::shared_ptr<Shape> m_shape;
    std::shared_ptr<Material> m_material;

    /** 
     * \note
     * - 0 : for radiance program
     * - 1 : for occlusion program (optional)
     **/
    std::vector<ProgramGroup> m_program_groups;
    // For managing sbt index which associated with a shader.
    uint32_t m_sbt_index { 0 };
};

// ---------------------------------------------------------------------
/** 
 * \brief
 * This class store the primitives with same transformation.
 * 
 * \note 
 * - Transform stored in this class must not be modified from outside.
 * - I decided to store primitives with the order of meshes -> custom primitves, 
 *   to set the sbt indices in the correct order.
 */
class PrimitiveInstance {
public:
    PrimitiveInstance() : m_transform(Transform()) {}
    explicit PrimitiveInstance(const sutil::Matrix4x4& mat) : m_transform(mat) {}
    explicit PrimitiveInstance(const Transform& transform) : m_transform(transform) {}
    explicit PrimitiveInstance(const Transform& transform, const std::vector<Primitive>& primitives)
    : m_transform(transform), m_primitives(primitives) {}

    void addPrimitive(const Primitive& p) { 
        m_primitives.push_back(p); 
        m_primitives.back().setSbtIndex(this->sbtIndexBase() + (this->numPrimitives() - 1));
    }
    void addPrimitive(Shape* shape, Material* mat_ptr) {
        m_primitives.emplace_back(shape, mat_ptr);
        m_primitives.back().setSbtIndex(this->sbtIndexBase() + (this->numPrimitives() - 1) );
    }

    /**
     * \brief Sort primitives with the order of meshes -> custom primitives.
     */
    void sort() {
        std::sort(m_primitives.begin(), m_primitives.end(), 
            [](const Primitive& p1, const Primitive& p2){ return (int)p1.shapeType() < (int)p2.shapeType(); });
        uint32_t sbt_index = 0;
        for (auto &p : m_primitives) {
            p.setSbtIndex(this->sbtIndexBase() + sbt_index);
            sbt_index++;
        }
    }

    // Allow to return primitives as lvalue. 
    std::vector<Primitive> primitives() const { return m_primitives; }
    std::vector<Primitive>& primitives() { return m_primitives; }

    size_t numPrimitives() const { return m_primitives.size(); }

    void setSbtIndexBase(const unsigned int base) { m_sbt_index_base = base; }
    unsigned int sbtIndexBase() const { return m_sbt_index_base; }
    unsigned int sbtIndex() const { return m_sbt_index_base + (unsigned int)m_primitives.size(); }
    
    void setTransform(const Transform& t) { m_transform = t; } 
    Transform transform() const { return m_transform; }
private:
    Transform m_transform;
    std::vector<Primitive> m_primitives;
    unsigned int m_sbt_index_base { 0 };
};

void buildGas(const OptixDeviceContext& ctx, AccelData& accel_data, const PrimitiveInstance& ps);

void buildInstances(const OptixDeviceContext& ctx, 
               const AccelData& accel_data,
               const PrimitiveInstance& primitive_instance, 
               unsigned int& sbt_base_offset,
               unsigned int& instance_id,
               std::vector<OptixInstance>& instances);

void createMaterialPrograms(
    const OptixDeviceContext& ctx,
    const Module& module, 
    std::vector<ProgramGroup>& program_groups, 
    std::vector<CallableRecord>& callable_records
);

void createTexturePrograms(
    const OptixDeviceContext& ctx, 
    const Module& module, 
    std::vector<ProgramGroup>& program_groups,
    std::vector<CallableRecord>& callable_records
);

}