#pragma once

#include "../core/util.h"
#include "../core/aabb.h"
#include "../optix/macros.h"
#include <sutil/vec_math.h>

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

struct AccelData {
    struct HandleData {
        OptixTraversableHandle handle { 0 };
        CUdeviceptr d_buffer { 0 };
        unsigned int count { 0 };
    };
    HandleData meshes {};
    HandleData customs {};

    ~AccelData() {
        if (meshes.d_buffer)  cudaFree( reinterpret_cast<void*>( meshes.d_buffer ) );
        if (customs.d_buffer) cudaFree( reinterpret_cast<void*>( customs.d_buffer ) );
    }
};

inline std::ostream& operator<<(std::ostream& out, const AccelData& accel) {
    out << "AccelData::meshes: " << accel.meshes.handle << ", " << accel.meshes.d_buffer << ", " << accel.meshes.count << std::endl;
    return out << "AccelData::customs: " << accel.customs.handle << ", " << accel.customs.d_buffer << ", " << accel.customs.count << std::endl;
}

class Shape {
public:
    virtual ~Shape() {}

    virtual OptixBuildInputType buildInputType() const = 0;
    virtual AABB bound() const = 0;

    virtual void prepareData() = 0;
    virtual void buildInput( OptixBuildInput& bi, uint32_t sbt_idx ) = 0;

    void attachSurface(const std::shared_ptr<Material>& material);
    void attachSurface(const std::shared_ptr<AreaEmitter>& area_emitter);
    void addProgram(const ProgramGroup& program);
    void free();

    template <class SBTRecord>
    void bindRecord(SBTRecord* record, int idx);

    void* devicePtr() const;
    std::vector<ProgramGroup> programs() const;
protected:
    void* d_data { 0 };
    CUdeviceptr d_aabb_buffer { 0 };

    std::vector<ProgramGroup> m_programs;
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> m_surface;
};

// Function definitions
void Shape::attachSurface(const std::shared_ptr<Material>& material)
{
    m_surface = material;
}

void Shape::attachSurface(const std::shared_ptr<AreaEmitter>& area_emitter)
{
    m_surface = area_emitter;
}

void Shape::addProgram(const ProgramGroup& program)
{
    if (program.kind() != OPTIX_PROGRAM_GROUP_KIND_HITGROUP)
    {
        Message(MSG_ERROR, "oprt::Shape::addProgram(): The kind of input program is not a OPTIX_PROGRAM_GROUP_KIND_HITGROUP.");
        return;
    }
    m_programs.push_back(program);
}

void Shape::free()
{
    if (d_aabb_buffer) cuda_free( d_aabb_buffer ); 
}

template <class SBTRecord>
void Shape::bindRecord(SBTRecord* record, int idx)
{
    if (m_programs.size() <= idx) {
        Message(MSG_ERROR, "oprt::Shape::bindRecord(): The index to bind SBT record exceeds the number of programs.");
        return;
    }
    m_programs[idx].bindRecord(record);
}

void* Shape::devicePtr() const
{
    return d_data;
}

std::vector<ProgramGroup> programs() const 
{
    return m_programs;
}

#endif // __CUDACC__

}