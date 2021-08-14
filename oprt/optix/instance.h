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

    explicit operator OptixInstance() const { return m_instance; }

    /** Update instance info on device */
    void update();

    void setId(const uint32_t id);
    void setSBTOffset(const uint32_t sbt_offset);
    void setVisibilityMask(const uint32_t visibility_mask);
    void setTraversableHandle(OptixTraversableHandle handle);
    void setPadding(const unsigned int pad[2]);

    /** Transformation of instance */
    void setTransform(const Transform& transform);
    void translate(const float3& t);
    void scale(const float3& scale);
    void scale(const float s);
    void rotate(const float radians, const float3& axis);
    Transform transform() const;

    /** Set flag */
    void disableTriangleFaceCulling();
    void flipTriangleFacing();
    void disableAnyhit();
    void enforceAnyhit();
    void disableTransform();

    bool isBuilded() const;

private:
    OptixInstance m_instance;
    CUdeviceptr d_instance;     // device pointer
    bool m_builded;        
};

}
