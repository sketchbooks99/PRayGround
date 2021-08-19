#pragma once 

#include <optix.h>
#include <oprt/optix/instance.h>
#include <oprt/optix/accel.h>
#include <oprt/core/shape.h>
#include <unordered_map>
#include <memory>

namespace oprt {

class ShapeInstance : public Instance {
public:
    enum class Type {
        Mesh = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
        Custom = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES,
        Curves = OPTIX_BUILD_INPUT_TYPE_CURVES, 
        None
    };

    ShapeInstance() = delete;
    explicit ShapeInstance(Type type);
    explicit ShapeInstance(Type type, const Transform& transform);
    explicit ShapeInstance(Type type, const Matrix4f& matrix);

    void copyToDevice() override;

    void buildAccelStructure();
    void updateAccelStructure();

    void setTraversableHandle(OptixTraversableHandle handle) override;

    void addShape(const std::string& name, const std::shared_ptr<Shape>& shape);
    std::shared_ptr<Shape> getShape(const std::string& name) const;
    uint32_t numShapes() const;

    Type type() const;

    OptixTraversableHandle handle();
private:
    GeometryAccel m_gas;
    Type m_type;
    std::unordered_map<std::string, std::shared_ptr<Shape>> m_shapes;
};

} // ::oprt
