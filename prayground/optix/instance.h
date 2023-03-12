#pragma once 

#include <prayground/core/util.h>
#include <prayground/core/shape.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/geometry_accel.h>
#include <vector>

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
        void translate(const Vec3f& t);
        void scale(const Vec3f& s);
        void scale(const float s);
        void rotate(const float radians, const Vec3f& axis);
        void rotateX(const float radians);
        void rotateY(const float radians);
        void rotateZ(const float radians);
        void reflect(Axis axis);
        Matrix4f transform();

        OptixInstance* rawInstancePtr() const;
    private:
        OptixInstance* m_instance;
    };

    /* 
    * ShapeInstance manages geometry acceleration structure (GAS), and instance properties.
    * This enables you to create instance acceleration structure (IAS) easily.
    */
    class ShapeInstance {
    public:
        ShapeInstance() = default;
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
        void translate(const Vec3f& t);
        void scale(const Vec3f& s);
        void scale(const float s);
        void rotate(const float radians, const Vec3f& axis);
        void rotateX(const float radians);
        void rotateY(const float radians);
        void rotateZ(const float radians);
        void reflect(Axis axis);
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
        Instance m_instance;
        GeometryAccel m_gas;
    };

} // namespace prayground
