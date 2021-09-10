#pragma once 

#include <prayground/core/util.h>
#include <prayground/core/shape.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/geometry_accel.h>
#include <vector>

namespace prayground {

/** 
 * OptixInstanceのWrapperクラス
*/
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
    OptixTraversableHandle handle() const;
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

/**
 * @note
 * TraversableHandleはGASをビルドしたときに自動で設定されるようにして
 * 外部から変更されないようにするためにInstanceを継承せずにメンバ変数として保持する
 */
class ShapeInstance {
public:
    ShapeInstance();
    ShapeInstance(ShapeType type);
    ShapeInstance(ShapeType type, const Matrix4f& m);
    ShapeInstance(ShapeType type, const std::shared_ptr<Shape>& shape);
    ShapeInstance(ShapeType type, const std::shared_ptr<Shape>& shape, const Matrix4f& m);

    explicit operator OptixInstance() const { return *(m_instance.rawInstancePtr()); }

    void addShape(const std::shared_ptr<Shape>& shape);
    std::vector<std::shared_ptr<Shape>> shapes() const;

    void setId(const uint32_t id);
    void setSBTOffset(const uint32_t sbt_offset);
    void setVisibilityMask(const uint32_t visibility_mask);
    void setPadding(uint32_t pad[2]);
    void setFlags(const uint32_t flags);

    uint32_t id() const;
    uint32_t sbtOffset() const;
    uint32_t visibilityMask() const;
    OptixTraversableHandle handle() const;
    uint32_t flags() const;

    void setTransform(const Matrix4f& matrix);
    void translate(const float3& t);
    void scale(const float3& s);
    void scale(const float s);
    void rotate(const float radians, const float3& axis);
    void rotateX(const float radians);
    void rotateY(const float radians);
    void rotateZ(const float radians);
    Matrix4f transform();

    void allowUpdate();
    void allowCompaction();
    void preferFastTrace();
    void preferFastBuild();
    void allowRandomVertexAccess();
    void buildAccel(const Context& ctx, CUstream stream);
    void updateAccel(const Context& ctx, CUstream stream);

    void destroy();

    OptixInstance* rawInstancePtr() const;
private:
    ShapeType m_type;
    GeometryAccel m_gas;
    Instance m_instance;
};

} // ::prayground
