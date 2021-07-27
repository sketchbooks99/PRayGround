/** 
 * @file primitive.h
 * @brief Management of primitives include shape and material.
 * @author Shunji Kiuchi
 * 
 * @details 
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
#include "../optix/context.h"
#include "../optix/program.h"
#include "../optix/sbt.h"
#include "../optix/module.h"
#include "../emitter/area.h"
#include <algorithm>

namespace oprt {

// APIs
void buildGas(const Context& ctx, AccelData& accel_data, const PrimitiveInstance& ps);

void buildInstances(const Context& ctx, 
               const AccelData& accel_data,
               const PrimitiveInstance& primitive_instance, 
               unsigned int& sbt_base_offset,
               unsigned int& instance_id,
               std::vector<OptixInstance>& instances);
    
AccelData oprtBuildGAS(const Context& ctx, const Primitive& p);
AccelData oprtBuildIAS(const Context& ctx, const Primitive& p, const Transform& transform);
OptixInstance oprtBuildInstance(const Context& ctx, const AccelData& accel, const Transform& transform);

void createMaterialPrograms(
    const Context& ctx,
    const Module& module, 
    std::vector<ProgramGroup>& program_groups, 
    std::vector<CallableRecord>& callable_records
);

void createTexturePrograms(
    const Context& ctx, 
    const Module& module, 
    std::vector<ProgramGroup>& program_groups,
    std::vector<CallableRecord>& callable_records
);

void createEmitterPrograms(
    const Context& ctx, 
    const Module& module, 
    std::vector<ProgramGroup>& program_groups, 
    std::vector<CallableRecord>& callable_records
);

/**
 * @class Primitive
 * @brief 
 * Primitive store an array of shape, array of material, 
 * and closest-hit program to describe the shape behavior on the device.
 */
class Primitive {
public:
    Primitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<Material>& material)
    : Primitive(shape, material, 0) {}

    Primitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<AreaEmitter>& area)
    : Primitive(shape, area, 0) {}

    Primitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<Material>& material, uint32_t sbt_index)
    : m_shape(shape), m_surface(material), m_sbt_index(sbt_index) 
    {
        m_program_groups.resize(RAY_TYPE_COUNT);
        for (auto &pg : m_program_groups) {
            pg = ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
        }
    }
    Primitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<AreaEmitter>& area, uint32_t sbt_index)
    : m_shape(shape), m_surface(area), m_sbt_index(sbt_index)
    {
        m_program_groups.resize(RAY_TYPE_COUNT);
        for (auto &pg : m_program_groups) {
            pg = ProgramGroup(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
        }
    }

    // Create programs based on shape type. 
    void createPrograms(const Context& ctx, const Module& module) {
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
    void prepareSurfaceData() 
    {
        std::visit([](auto& surface) { surface->prepareData(); }, m_surface);
    }

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
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> surface() const { return m_surface; }
    std::shared_ptr<Shape> shape() const { return m_shape; }
    ShapeType shapeType() const { return m_shape->type(); }
    SurfaceType surfaceType() const {
        if (std::holds_alternative<std::shared_ptr<Material>>(m_surface))
            return SurfaceType::Material;
        else 
            return SurfaceType::Emitter;
    }

    std::vector<ProgramGroup> programGroups() const { return m_program_groups; }

private:
    // Member variables.
    std::shared_ptr<Shape> m_shape;
    // std::shared_ptr<Material> m_material;
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> m_surface;

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
 * @brief
 * This class store the primitives with same transformation.
 * 
 * @note 
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
        m_primitives.back().setSbtIndex(sbtIndexBase() + (numPrimitives() - 1));
    }
    void addPrimitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<Material>& material) {
        m_primitives.emplace_back(shape, material);
        m_primitives.back().setSbtIndex(sbtIndexBase() + (numPrimitives() - 1) );
    }
    void addPrimitive(const std::shared_ptr<Shape>& shape, const std::shared_ptr<AreaEmitter>& area) {
        m_primitives.emplace_back(shape, area);
        m_primitives.back().setSbtIndex(sbtIndexBase() + (numPrimitives() - 1) );
    }

    /**
     * \brief Sort primitives with the order of meshes -> custom primitives.
     */
    void sort() {
        std::sort(m_primitives.begin(), m_primitives.end(), 
            [](const Primitive& p1, const Primitive& p2){ return (int)p1.shapeType() < (int)p2.shapeType(); });
        uint32_t sbt_index = 0;
        for (auto &p : m_primitives) {
            p.setSbtIndex(sbtIndexBase() + sbt_index);
            sbt_index++;
        }
    }

    // Allow to return primitives as lvalue. 
    std::vector<Primitive> primitives() const { return m_primitives; }
    std::vector<Primitive>& primitives() { return m_primitives; }

    size_t numPrimitives() const { return m_primitives.size(); }

    void setSbtIndexBase(const unsigned int base) { m_sbt_index_base = base; }
    unsigned int sbtIndexBase() const { return m_sbt_index_base; }
    unsigned int sbtIndex() const { return m_sbt_index_base + static_cast<unsigned int>(m_primitives.size()); }
    
    void setTransform(const Transform& t) { m_transform = t; }
    Transform transform() const { return m_transform; }
private:
    Transform m_transform;
    std::vector<Primitive> m_primitives;
    unsigned int m_sbt_index_base { 0 };
};

}