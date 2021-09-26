#pragma once

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#include <prayground/core/cudabuffer.h>
#endif

namespace prayground {

struct CurvesData
{

};

#ifndef __CUDACC__
class Curves final : public Shape {
public:
    using DataType = CurvesData;

    Curves();

    OptixBuildInputType buildInputType() const override;
    void copyToDevice() override;
    void buildInput(OptixBuildInput& bi) override;
    AABB bound() const override;
private:

};

}