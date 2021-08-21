#pragma once

#include <variant>
#include <memory>
#include <oprt/core/util.h>
#include <oprt/core/aabb.h>
#include <oprt/core/material.h>
#include <oprt/core/interaction.h>
#include <oprt/emitter/area.h>
#include <oprt/optix/macros.h>
#include <oprt/optix/program.h>
#include <sutil/vec_math.h>
#include <optix_types.h>

namespace oprt {

enum class ShapeType {
    Mesh = 0,        
    Sphere = 1,      
    Plane = 2,       
    Cylinder = 3, 
    Arbitrary = 4
};

#ifndef __CUDACC__

/** 
 * \brief 
 * Map object to easily get string of shape via ShapeType, 
 * ex) 
 *  const char* shape_str = shape_map[ShapeType::Mesh] -> "mesh"
 **/

/**
 * @todo
 * Enum and maps associated with shape type will be deprecated.
 */
static std::map<ShapeType, const char*> shape_map = {
    { ShapeType::Mesh, "mesh" },
    { ShapeType::Sphere, "sphere" },
    { ShapeType::Plane, "plane" }, 
    { ShapeType::Cylinder, "cylinder" }
};

static std::map<ShapeType, const char*> shape_occlusion_map = {
    { ShapeType::Mesh, "mesh_occlusion"},
    { ShapeType::Sphere, "sphere_occlusion"},
    { ShapeType::Plane, "plane_occlusion" }, 
    { ShapeType::Cylinder, "cylinder_occlusion" }
};

inline std::ostream& operator<<(std::ostream& out, ShapeType type) {
    switch(type) {
    case ShapeType::Mesh:       return out << "ShapeType::Mesh";
    case ShapeType::Sphere:     return out << "ShapeType::Sphere";
    case ShapeType::Plane:      return out << "ShapeType::Plane";
    case ShapeType::Cylinder:   return out << "ShapeType::Cylinder";
    default:                    return out << "";
    }
}

class Shape {
public:
    virtual ~Shape() {}

    virtual OptixBuildInputType buildInputType() const = 0;
    virtual AABB bound() const = 0;

    virtual void copyToDevice() = 0;
    virtual void buildInput( OptixBuildInput& bi ) = 0;

    void attachSurface(const std::shared_ptr<Material>& material);
    void attachSurface(const std::shared_ptr<AreaEmitter>& area_emitter);
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> surface() const;
    SurfaceType surfaceType() const;
    void* surfaceDevicePtr() const;

    void addProgram(const ProgramGroup& program);
    std::vector<ProgramGroup> programs() const;
    ProgramGroup programAt(int idx) const;

    void setSbtIndex(const uint32_t sbt_index);
    uint32_t sbtIndex() const;

    void free();
    void freeAabbBuffer();
    
    void* devicePtr() const;

    template <class SBTRecord>
    void bindRecord(SBTRecord* record, int idx)
    {
        if (m_programs.size() <= idx) {
            Message(MSG_ERROR, "oprt::Shape::bindRecord(): The index to bind SBT record exceeds the number of programs.");
            return;
        }
        m_programs[idx]->bindRecord(record);
    }

protected:
    void* d_data { 0 };
    CUdeviceptr d_aabb_buffer { 0 };
    uint32_t m_sbt_index;

private:
    std::vector<std::unique_ptr<ProgramGroup>> m_programs;
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> m_surface;
};

#endif // __CUDACC__

}