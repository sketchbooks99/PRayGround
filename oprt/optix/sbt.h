#pragma once

#include <optix.h>
#include <concepts>
#include <type_traits>

namespace oprt {

/**
 * @note 
 * Should I implement a wrapper for Optix Shader Binding Table ??
 */

namespace {
    template <class T>
    concept TrivialCopyable = std::is_trivially_copyable_v<T>;

    template <class Head, class... Args>
    void push_to_vector(std::vector<Head>& v, const Head& head, const Args&... args)
    {
        v.emplace_back(head);
        if constexpr (sizeof...(args) != 0)
            push_to_vector(v, args...);
    }
} // ::nonamed namespace

#ifndef __CUDACC__
template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

/**
 * @note 
 * 各種データが関数を保持した構造体だと、デバイス上におけるインスタンス周りで Invalid access
 * が生じる可能性があるのでそこは <concepts> 等で制約をかけたい
 * 
 * SPECIALTHANKS:
 * - @yaito3014 gives an advise that to use is_trivially_copyable_v 
 *   for prohibiting declaration of virtual functions in SBT Data.
 */

template <TrivialCopyable RayGenData, TrivialCopyable MissData, TrivialCopyable HitGroupData, 
          TrivialCopyable CallablesData, TrivialCopyable ExceptionData, size_t NRay>
class ShaderBindingTable {
public:
    ShaderBindingTable()
    {
        static_assert(NRay > 0, "The number of ray types must be at least 1.");
    }

    explicit operator OptixShaderBindingTable() const { return m_sbt; }

    void setRaygenData(const RayGenData& rg_data)
    {
        m_raygen_data = rg_data;
    }

    template <class... MissDataArgs>
    void setMissData(const MissDataArgs&... args)
    {
        static_assert(sizeof...(args) == N, 
            "oprt::ShaderBindingTable::addMissData(): The number of data must be same with the number of ray types.");        
        static_assert(conjunction<is_same<T, Args>...>::value, "oprt::ShaderBindingTable::addMissData(): Data type must be same with 'MissData'.");

        if (!m_miss_datas.empty()) m_miss_datas.clear();
        push_to_vector(m_miss_datas, args...);
    }

    template <class... HitGroupDataArgs> 
    void addHitgroupData(const HitGroupDataArgs...& args)
    {
        static_assert(sizeof...(args) == N, 
            "oprt::ShaderBindingTable::addHitgroupData(): The number of hitgroup data must be same with the number of ray types.");        
        static_assert(conjunction<is_same<T, Args>...>::value, "oprt::ShaderBindingTable::addHitgroupData(): Data type must be same with 'HitgroupData'.");

        push_to_vector(m_hitgroup_datas, args...);
    }

    template <class... CallablesDataArgs>
    void addCallablesData(const CallablesDataArgs&... args)
    {
        static_assert(conjunction<is_same<T, Args>...>::value, "oprt::ShaderBindingTable::addCallablesData(): Data type must be same with 'CallablesData'.");

        push_to_vector(m_callables_datas, args...);
    }

    void setExceptionData(const ExceptionData& ex_data)
    {
        m_exception_data = ex_data;
    }

    void createOnDevice()
    {

    }

    OptixShaderBindingTable sbt() const 
    {
        return sbt;
    }
private:
    OptixShaderBindingTable m_sbt {};
    RayGenData m_raygen_data;
    std::vector<MissData> m_miss_datas;
    std::vector<HitGroupData> m_hitgroup_datas;
    std::vector<CallablesData> m_callables_datas;
    ExceptionData m_exception_data;
};

#endif

}