#pragma once 

#include <prayground/optix/sbt.h>
#include <prayground/optix/pipeline.h>
#include <prayground/optix/accel.h>
#include <type_traits>
#include <unordered_map>

namespace prayground {

namespace {
    // OptiXが仮想関数を許可していないので、Shader binding tableの各種データが
    // 仮想関数を持たないように制限をかける
    template <class T>
    concept TrivialCopyable = std::is_trivially_copyable_v<T>;
}

template <TrivialCopyable RaygenData, TrivialCopyable MissData, TrivialCopyable HitgroupData,
          TrivialCopyable CallablesData, TrivialCopyable ExceptionData, unsigned int NRay>
class RTScene {
public:
    RTScene(){}

    void setPipeline(const Pipeline& pipeline)
    {
        m_pipeline = pipeline;
    }

    void buildAccel(const Context& ctx, CUstream cu_stream)
    {

    }

    void updateAccel(const Context& ctx, CUstream cu_stream)
    {

    }

    OptixTravresableHandle accelHandle() const
    {
        return ias.handle();
    }

    void createSBTOnDevice()
private:
    InstanceAccel ias;
    // ASのトップから見える1次のインスタンスのみを管理
    // それ以上の階層はアプリケーション側で管理する
    // 外部でInstanceの値に修正が加わってもSceneのインスタンス情報も更新されるように
    // 共有ポインタで管理しておく
    std::vector<std::shared_ptr<Instance>> m_instances;
    Pipeline m_pipeline;
    ShaderBindingTable<Record<RaygenData>, Record<MissData>, Record<HitgroupData>, Record<CallablesData>, Record<ExceptionData>, NRay> m_sbt;
};

} // ::prayground