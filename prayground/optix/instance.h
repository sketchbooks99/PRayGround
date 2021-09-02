#pragma once 

#include <prayground/core/util.h>
#include <prayground/core/shape.h>
#include <prayground/core/transform.h>
#include <unordered_map>

/** OptixInstanceのラッパー 
 * 
*/

namespace prayground {

class Instance {
public:
    Instance();
    Instance(const Matrix4f& matrix);
    Instance(const Instance& instance);

    explicit operator OptixInstance() const { return *m_instance; }

    void setId(const uint32_t id);
    void setSBTOffset(const uint32_t sbt_offset);
    void setVisibilityMask(const uint32_t visibility_mask);
    // Instance の TraversableHandle を変更する際は ASは更新ではなく再ビルドする必要がある
    void setTraversableHandle(OptixTraversableHandle handle);
    void setPadding(uint32_t pad[2]);
    void setFlags(const uint32_t flags);

    uint32_t id() const;
    uint32_t sbtOffset() const;
    uint32_t visibilityMask() const;
    OptixTraversableHandle handle();
    uint32_t flags() const;

    /** Transformation of instance */
    void setTransform(const Matrix4f& matrix);
    void translate(const float3& t);
    void scale(const float3& s);
    void scale(const float s);
    void rotate(const float radians, const float3& axis);
    void rotateX(const float radians);
    void rotateY(const float radians);
    void rotateZ(const float radians);
    Matrix4f transform();

    OptixInstance* rawInstancePtr() const;
private:
    /**
     * @note 
     * InstanceAccelの内部で管理しているInstanceを外部から間接的に更新できるように
     * OptixInstance はポインタで管理しておく
     * shared_ptr<Instance> にしてm_instance を実体にしておくかは悩みどころ
     */
    OptixInstance* m_instance;
};

} // ::prayground
