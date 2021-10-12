#pragma once

#include <optix.h>
#include <prayground/core/camera.h>

#ifndef __CUDACC__
    #include <concepts>
    #include <type_traits>
    #include <prayground/core/cudabuffer.h>
#endif

namespace prayground {

/**
 * デフォルトのレイトレ実装 <prayground/optix/cuda/prayground.cu> などが動くようにビルトインパラメータを宣言しておく
 */

struct pgLaunchParams 
{
    unsigned int width, height;
    unsigned int samples_per_launch;
    int subframe_index;
    float4* accum_buffer;
    uchar4* result_buffer;
    OptixTraversableHandle handle;
};

struct pgRaygenData
{
    builtin::CameraData camera;
};

struct pgHitgroupData
{
    void* shape_data;
    void* surface_data;

    unsigned int surface_program_id;
    unsigned int surface_pdf_id;
};

struct pgMissData
{
    void* env_data;
};

struct pgEmptyData
{

};

#ifndef __CUDACC__

namespace {
    template <class T>
    concept TrivialCopyable = std::is_trivially_copyable_v<T>;

    template <class T>
    concept HasData = requires(T t)
    {
        t.data;
    };

    template <class Head, class... Args>
    inline void push_to_vector(std::vector<Head>& v, const Head& head, const Args&... args)
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

template <HasData RaygenRecord, HasData MissRecord, HasData HitgroupRecord, 
          HasData CallablesRecord, HasData ExceptionRecord, uint32_t N>
class ShaderBindingTable {
public:
    static constexpr uint32_t NRay = N;

    ShaderBindingTable()
    {
        static_assert(N > 0, "The number of ray types must be at least 1.");
    }

    explicit operator OptixShaderBindingTable() const { return m_sbt; }

    void setRaygenRecord(const RaygenRecord& rg_record)
    {
        m_raygen_record = rg_record;
    }
    CUdeviceptr raygenRecord() const 
    {
        return m_sbt.raygenRecord;
    }

    template <class... MissRecordArgs>
    void setMissRecord(const MissRecordArgs&... args)
    {
        static_assert(sizeof...(args) == N, 
            "prayground::ShaderBindingTable::setMissRecord(): The number of record must be same with the number of ray types.");        
        static_assert(std::conjunction<std::is_same<MissRecord, MissRecordArgs>...>::value, 
            "prayground::ShaderBindingTable::setMissRecord(): Data type must be same with 'MissRecord'.");

        push_to_vector(m_miss_records, args...);
    }



    /// @note 置き換えを行ったらデバイス上のデータも更新する？
    void replaceMissRecord(const MissRecord& record, const int idx)
    {
        if (idx >= m_miss_records.size())
        {
            Message(MSG_ERROR, "prayground::ShaderBindingTable::replaceMissRecord(): The index out of range.");
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
        static_assert(sizeof...(args) == N, 
            "prayground::ShaderBindingTable::addHitgroupRecord(): The number of hitgroup record must be same with the number of ray types.");        
        static_assert(std::conjunction<std::is_same<HitgroupRecord, HitgroupRecordArgs>...>::value, 
            "prayground::ShaderBindingTable::addHitgroupRecord(): Record type must be same with 'HitgroupRecord'.");
        push_to_vector(m_hitgroup_records, args...);
    }

    /// @note 置き換えを行ったらデバイス上のデータも更新する？
    void replaceHitgroupRecord(HitgroupRecord record, const int idx)
    {
        if (idx >= m_hitgroup_records.size())
        {
            Message(MSG_ERROR, "prayground::ShaderBindingTable::replaceHitgroupRecord(): The index out of range.");
            return;
        }
        m_hitgroup_records[idx] = record;

        if (m_sbt.hitgroupRecordBase)
        {
            HitgroupRecord* hg_ptr = &reinterpret_cast<HitgroupRecord*>(m_sbt.hitgroupRecordBase)[idx];
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(&hg_ptr->data), 
                &record.data, sizeof(record.data), cudaMemcpyHostToDevice
            ));
        }
    }

    template <class... CallablesRecordArgs>
    void addCallablesRecord(const CallablesRecordArgs&... args)
    {
        static_assert(std::conjunction<std::is_same<CallablesRecord, CallablesRecordArgs>...>::value, 
            "prayground::ShaderBindingTable::addCallablesRecord(): Record type must be same with 'CallablesRecord'.");

        push_to_vector(m_callables_records, args...);
    }

    /// @note 置き換えを行ったらデバイス上のデータも更新する？
    void replaceCallablesRecord(const CallablesRecord& record, const int idx)
    {
        if (idx >= m_callables_records.size())
        {
            Message(MSG_ERROR, "prayground::ShaderBindingTable::replaceCallablesRecord(): The index out of range.");
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
        d_miss_records.copyToDevice(m_miss_records);
        d_hitgroup_records.copyToDevice(m_hitgroup_records);
        d_callables_records.copyToDevice(m_callables_records);
        d_exception_record.copyToDevice(&m_exception_record, sizeof(ExceptionRecord));

        m_sbt.raygenRecord = d_raygen_record.devicePtr();
        m_sbt.missRecordBase = d_miss_records.devicePtr();
        m_sbt.missRecordCount = static_cast<uint32_t>(m_miss_records.size());
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
    std::vector<MissRecord> m_miss_records;
    std::vector<HitgroupRecord> m_hitgroup_records;
    std::vector<CallablesRecord> m_callables_records;
    ExceptionRecord m_exception_record;
    bool on_device;
};

#endif // __CUDACC__

} // ::prayground