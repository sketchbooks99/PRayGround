#include "shape_instance.h"

namespace oprt {

// ------------------------------------------------------------------
ShapeInstance::ShapeInstance(Type type)
    : m_type(type), Instance()
{

}

ShapeInstance::ShapeInstance(Type type, const Matrix4f& matrix)
    : m_type(type), Instance(matrix)
{

}

ShapeInstance::ShapeInstance(Type type, const Transform& transform)
{

}

// ------------------------------------------------------------------
void ShapeInstance::copyToDevice()
{
    for (auto& shape : m_shapes) {
        if (!shape.second->devicePtr())
            shape.second->copyToDevice();
    }
    Instance::copyToDevice();
}

// ------------------------------------------------------------------
void ShapeInstance::buildAccelStructure()
{
    
}

void ShapeInstance::updateAccelStructure()
{

}

// ------------------------------------------------------------------
void ShapeInstance::setTraversableHandle(OptixTraversableHandle handle)
{
    
}

} // ::oprt