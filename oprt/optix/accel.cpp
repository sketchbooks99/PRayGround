#include "accel.h"

namespace oprt {

// GeometryAccel -------------------------------------------------------------
GeometryAccel::GeometryAccel()
{

}

GeometryAccel(const OptixAccelBuildOptions& options)
: m_options(options)
{

}

GeometryAccel::~GeometryAccel()
{

}

// ---------------------------------------------------------------------------
void GeometryAccel::build(const Context& ctx, const std::shared_ptr<Shape>& shape)
{

}

void GeometryAccel::build(const Context& ctx, const std::vector<std::shared_ptr<Shape>>& shapes)
{

}

void GeometryAccel::free()
{

}

// ---------------------------------------------------------------------------
uint32_t GeometryAccel::count() const
{
    return m_count;
}

OptixTraversableHandle GeometryAccel::handle() const
{
    return m_handle;
}

// InstanceAccel -------------------------------------------------------------
InstanceAccel::InstanceAccel()
{

}

InstanceAccel::InstanceAccel(const OptixAccelBuildOptions& options)
: m_options(options)
{

}

// ---------------------------------------------------------------------------
void InstanceAccel::build(const Context& ctx, const Instance& instace)
{

}

void InstanceAccel::build(const Context& ctx, const OptixInstance& instance)
{

}

void InstanceAccel::build(const Context& ctx, const std::vector<Instance>& instances)
{

}

void InstanceAccel::build(const Context& ctx, const std::vector<OptixInstance>& optix_instances)
{

}

void InstanceAccel::free()
{

}

// ---------------------------------------------------------------------------
uint32_t InstanceAccel::count() const
{
    return m_count;
}

OptixTraversableHandle InstanceAccel::handle() const
{
    return m_handle;
}

} // ::oprt