#pragma once 

#include "../core/util.h"
#include "../core/shape.h"
#include "../core/transform.h"
#include "util.h"

/**
 * @todo
 * OptixInstanceとそれによるIASのビルドに対するラッパーの実装
 */

namespace oprt {

class Instance {
public:
    Instance();
    Instance(const Transform& transform);

    explicit operator OptixInstance() const { return m_instance; }

    /** Update instance info on device */
    void copyToDevice();

    void setId(const uint32_t id);
    void setSBTOffset(const uint32_t sbt_offset);
    void setVisibilityMask(const uint32_t visibility_mask);
    void setTraversableHandle(OptixTraversableHandle handle);
    void setPadding(uint32_t pad[2]);
    void setFlags(const uint32_t flags);

    /** Transformation of instance */
    void setTransform(const Transform& transform);
    void setTransform(const Matrix4f& matrix);
    void translate(const float3& t);
    void scale(const float3& s);
    void scale(const float s);
    void rotate(const float radians, const float3& axis);
    void rotateX(const float radians);
    void rotateY(const float radians);
    void rotateZ(const float radians);
    Transform transform() const;

    bool isDataOnDevice() const;
    CUdeviceptr devicePtr() const;
private:
    OptixInstance m_instance;
    CUdeviceptr d_instance;
};

} // ::oprt
