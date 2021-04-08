#pragma once

#include <include/core/util.h>
#include <include/core/aabb.h>
#include <include/optix/macros.h>
#include <sutil/vec_math.h>
#include <map>

namespace pt {

enum class ShapeType {
    None,       // None type
    Mesh,       // Mesh with triangle 
    Sphere,     // Sphere 
    Plane       // Plane (rectangle)
};

/** 
 * \brief 
 * Map object to easily get string of shape via ShapeType, 
 * ex) 
 *  const char* shape_str = shape_map[ShapeType::Mesh] -> "mesh"
 **/
static std::map<ShapeType, const char*> shape_map = {
    {ShapeType::Mesh, "mesh"},
    {ShapeType::Sphere, "sphere"}
};

inline std::ostream& operator<<(std::ostream& out, ShapeType type) {
    switch(type) {
    case ShapeType::None:
        return out << "ShapeType::None";
    case ShapeType::Mesh:
        return out << "ShapeType::Mesh";
    case ShapeType::Sphere:
        return out << "ShapeType::Sphere";
    case ShapeType::Plane:
        return out << "ShapeType::Plane";
    default:
        return out << "";
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
    virtual ShapeType type() const = 0;
    virtual AABB bound() const = 0;

    virtual void prepare_data() = 0;
    virtual void build_input( OptixBuildInput& bi, uint32_t sbt_idx, unsigned int index_offset ) = 0;
    void free_aabb_buffer() {
        if (d_aabb_buffer) CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aabb_buffer))); 
    }

    CUdeviceptr get_dptr() const { return d_data; }
    CUdeviceptr& get_dptr() { return d_data; }
protected:
    CUdeviceptr d_data { 0 };
    CUdeviceptr d_aabb_buffer { 0 };
};

}