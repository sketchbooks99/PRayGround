#pragma once

#include <optix.h>

#ifndef __CUDACC__
    #include <concepts>
    #include <type_traits>
    #include <oprt/core/cudabuffer.h>
#endif

namespace oprt {

/**
 * デフォルトのレイトレ実装 <oprt/optix/cuda/oprt.cu> などが動くようにビルトインパラメータを宣言しておく
 */
namespace builtin {

struct LaunchParams 
{
    unsigned int width, height;
    unsigned int samples_per_launch;
    int subframe_index;
    uchar4* result_buffer;
    OptixTraversableHandle handle;
};

struct CameraData
{
    float3 origin; 
    float3 lookat; 
    float3 up;
    float fov;
    float aspect;
};

struct RaygenData
{
    CameraData camera;
};

struct HitgroupData
{
    void* shape_data;
    void* surface_data;

    unsigned int surface_program_id;
    unsigned int surface_pdf_id;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};

} // ::builtin

#ifndef __CUDACC__

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

template <TrivialCopyable T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

/**
 * @note 
 * 各種データが仮想関数を保持したしている場合、デバイス上におけるインスタンス周りで 
 * Invalid accessが生じる可能性があるので <concepts> 等で制約をかけたい
 * 
 * SPECIALTHANKS:
 * @yaito3014 gave me an advise that to use is_trivially_copyable_v 
 * for prohibiting declaration of virtual functions in SBT Data.
 */

template <class RaygenRecord, class MissRecord, class HitgroupRecord, 
          class CallablesRecord, class ExceptionRecord, unsigned int NRay>
class ShaderBindingTable {
public:
    ShaderBindingTable()
    {
        static_assert(NRay > 0, "The number of ray types must be at least 1.");
    }

    explicit operator OptixShaderBindingTable() const { return m_sbt; }

    void setRaygenRecord(const RaygenRecord& rg_record)
    {
        m_raygen_record = rg_record;
    }

    template <class... MissRecordArgs>
    void setMissRecord(const MissRecordArgs&... args)
    {
        static_assert(sizeof...(args) == NRay, 
            "oprt::ShaderBindingTable::addMissRecord(): The number of record must be same with the number of ray types.");        
        static_assert(std::conjunction<std::is_same<MissRecord, MissRecordArgs>...>::value, 
            "oprt::ShaderBindingTable::addMissRecord(): Data type must be same with 'MissRecord'.");

        int j = 0;
        for (auto i : std::initializer_list<std::common_type_t<MissRecordArgs...>>{args...})
            m_miss_records[j++] = i;
    }

    /// @note 置き換えを行ったらデバイス上のデータも更新する？
    void replaceMissRecord(const MissRecord& record, const int idx)
    {
        if (idx >= m_miss_records.size())
        {
            Message(MSG_ERROR, "oprt::ShaderBindingTable::replaceMissRecord(): The index out of range.");
            return;
        }
        m_miss_records[idx] = record;
    }

    MissRecord getMissRecord(const int idx) const {
        return m_miss_records[idx];
    }

    template <class... HitgroupRecordArgs> 
    void addHitgroupRecord(const HitgroupRecordArgs&... args)
    {
        static_assert(sizeof...(args) == NRay, 
            "oprt::ShaderBindingTable::addHitgroupRecord(): The number of hitgroup record must be same with the number of ray types.");        
        static_assert(std::conjunction<std::is_same<HitgroupRecord, HitgroupRecordArgs>...>::value, 
            "oprt::ShaderBindingTable::addHitgroupRecord(): Record type must be same with 'HitgroupRecord'.");
        push_to_vector(m_hitgroup_records, args...);
    }

    /// @note 置き換えを行ったらデバイス上のデータも更新する？
    void replaceHitgroupRecord(const HitgroupRecord& record, const int idx)
    {
        if (idx >= m_hitgroup_records.size())
        {
            Message(MSG_ERROR, "oprt::ShaderBindingTable::replaceHitgroupRecord(): The index out of range.");
            return;
        }
        m_hitgroup_records[idx] = record;
    }

    template <class... CallablesRecordArgs>
    void addCallablesRecord(const CallablesRecordArgs&... args)
    {
        static_assert(std::conjunction<std::is_same<CallablesRecord, CallablesRecordArgs>...>::value, 
            "oprt::ShaderBindingTable::addCallablesRecord(): Record type must be same with 'CallablesRecord'.");

        push_to_vector(m_callables_records, args...);
    }

    /// @note 置き換えを行ったらデバイス上のデータも更新する？
    void replaceCallablesRecord(const CallablesRecord& record, const int idx)
    {
        if (idx >= m_callables_records.size())
        {
            Message(MSG_ERROR, "oprt::ShaderBindingTable::replaceCallablesRecord(): The index out of range.");
            return;
        }
        m_callables_records[idx] = record;
    }

    void setExceptionRecord(const ExceptionRecord& ex_record)
    {
        m_exception_record = ex_record;
    }

    void createOnDevice()
    {
        CUDABuffer<RaygenRecord> d_raygen_record;
        CUDABuffer<MissRecord> d_miss_records;
        CUDABuffer<HitgroupRecord> d_hitgroup_records;
        CUDABuffer<CallablesRecord> d_callables_records;
        CUDABuffer<ExceptionRecord> d_exception_record;

        d_raygen_record.copyToDevice(&m_raygen_record, sizeof(RaygenRecord));
        d_miss_records.copyToDevice(m_miss_records, NRay * sizeof(MissRecord));
        d_hitgroup_records.copyToDevice(m_hitgroup_records.data(), m_hitgroup_records.size() * sizeof(HitgroupRecord));
        d_callables_records.copyToDevice(m_callables_records.data(), m_callables_records.size() * sizeof(CallablesRecord));
        d_exception_record.copyToDevice(&m_exception_record, sizeof(ExceptionRecord));

        m_sbt.raygenRecord = d_raygen_record.devicePtr();
        m_sbt.missRecordBase = d_miss_records.devicePtr();
        m_sbt.missRecordCount = NRay;
        m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissRecord));
        m_sbt.hitgroupRecordBase = d_hitgroup_records.devicePtr();
        m_sbt.hitgroupRecordCount = static_cast<uint32_t>(m_hitgroup_records.size());
        m_sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitgroupRecord));
        m_sbt.callablesRecordBase = d_callables_records.devicePtr();
        m_sbt.callablesRecordCount = static_cast<uint32_t>(m_callables_records.size());
        m_sbt.callablesRecordStrideInBytes = static_cast<uint32_t>(sizeof(CallablesRecord));
        m_sbt.exceptionRecord = d_exception_record.devicePtr();

        on_device = true;
    }

    OptixShaderBindingTable& sbt()
    {
        return m_sbt;
    }

    bool isOnDevice() const 
    {
        return on_device;
    }
private:
    OptixShaderBindingTable m_sbt {};
    RaygenRecord m_raygen_record;
    MissRecord m_miss_records[NRay];
    std::vector<HitgroupRecord> m_hitgroup_records;
    std::vector<CallablesRecord> m_callables_records;
    ExceptionRecord m_exception_record;
    bool on_device;
};

#endif // __CUDACC__

}