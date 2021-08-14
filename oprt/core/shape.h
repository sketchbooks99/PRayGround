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

// Abstract class for readability
class Shape {
public:
    virtual ~Shape() {}

    virtual ShapeType type() const = 0;
    virtual AABB bound() const = 0;

    virtual void prepareData() = 0;
    virtual void buildInput( OptixBuildInput& bi, uint32_t sbt_idx ) = 0;
    void freeAabbBuffer() {
        if (d_aabb_buffer) cuda_free( d_aabb_buffer ); 
    }

    /// @todo
    void attachSurface(const std::shared_ptr<Material>& mat_ptr);
    void addProgram(const ProgramGroup& program);

    /// @todo
    void translate(const float3& t);
    void scale(const float3& scale);
    void scale(const float s);
    void rotate(const float radians, const float3& axis);

    void* devicePtr() const { return d_data; }
protected:
    void* d_data { 0 };
    CUdeviceptr d_aabb_buffer { 0 };

    /// @todo 
    Transform m_transform;
    std::shared_ptr<ProgramGroup> m_programs;
    std::variant<std::shared_ptr<Material>, std::shared_ptr<AreaEmitter>> m_surface;
};

#endif // __CUDACC__

}