#pragma once

#include <optix.h>
#include <prayground/core/camera.h>
#include <prayground/core/interaction.h>

#ifndef __CUDACC__
#include <concepts>
#include <type_traits>
#include <prayground/core/cudabuffer.h>
#endif // __CUDACC__

namespace prayground {

    enum class SBTRecordType : uint32_t {
        None = 0, 
        Raygen = 1u << 0,
        Miss = 1u << 1,
        Hitgroup = 1u << 2, 
        Callables = 1u << 3, 
        Exception = 1u << 4
    };

    constexpr SBTRecordType  operator|(SBTRecordType t1, SBTRecordType t2)  { return static_cast<SBTRecordType>((uint32_t)t1 | (uint32_t)t2); }
    constexpr SBTRecordType  operator|(uint32_t t1, SBTRecordType t2)       { return static_cast<SBTRecordType>(          t1 | (uint32_t)t2); }
    constexpr SBTRecordType  operator&(SBTRecordType t1, SBTRecordType t2)  { return static_cast<SBTRecordType>((uint32_t)t1 & (uint32_t)t2); }
    constexpr SBTRecordType  operator&(uint32_t t1, SBTRecordType t2)       { return static_cast<SBTRecordType>(          t1 & (uint32_t)t2); }
    constexpr SBTRecordType  operator~(SBTRecordType t1)                    { return static_cast<SBTRecordType>(~(uint32_t)t1); }
    constexpr uint32_t       operator+(SBTRecordType t1)                    { return static_cast<uint32_t>(t1); }

    /* Default structs for shader binding table records. */
    template <class Cam>
    struct pgRaygenData {
        typename Cam::Data camera;
    };

    struct pgHitgroupData {
        void* shape_data;
        SurfaceInfo surface_info;
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
        concept HasData = requires(T t)
        {
            t.data;
        };
    } // nonamed namespace

    template <class T>
    struct Record 
    {
        __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    template <HasData RaygenRecord, HasData MissRecord, HasData HitgroupRecord, 
              HasData CallablesRecord, HasData ExceptionRecord, uint32_t N>
    requires (N > 0)
    class ShaderBindingTable {
    public:
        static constexpr uint32_t NRay = N;

        ShaderBindingTable() {}

        explicit operator OptixShaderBindingTable() const { return m_sbt; }

        // ------------------------------------------------------------------
        // Raygen
        // ------------------------------------------------------------------
        void setRaygenRecord(const RaygenRecord& rg_record) 
        {
            m_raygen_record = rg_record;
        }
        RaygenRecord& raygenRecord() 
        {
            return m_raygen_record;
        }
        const RaygenRecord& raygenRecord() const 
        {
            return m_raygen_record;
        }
        CUdeviceptr deviceRaygenRecordPtr() const 
        {
            if (m_sbt.raygenRecord)
                return m_sbt.raygenRecord;
            else
            {
                THROW("Shader binding table hasn't been created on device yet.");
                return 0ull;
            }
        }

        // ------------------------------------------------------------------
        // Raygen
        // ------------------------------------------------------------------
        void setMissRecord(const std::array<MissRecord, N>& miss_records)
        {
            m_miss_records = miss_records;
        }

        /// @note 置き換えを行ったらデバイス上のデータも更新する？
        void replaceMissRecord(const MissRecord& record, const int idx) 
        {
            if (idx >= N)
            {
                pgLogFatal("The index out of range");
                return;
            }
            m_miss_records[idx] = record;
        }
        MissRecord& missRecord(const int idx) 
        {
            return m_miss_records[idx];
        }
        const MissRecord& missRecord(const int idx) const 
        {
            return m_miss_records[idx];
        }
        CUdeviceptr deviceMissRecordPtr() const 
        {
            if (m_sbt.missRecordBase)
                return m_sbt.missRecordBase;
            else
            {
                THROW("Shader binding table hasn't been created on device yet.");
                return 0ull;
            }
        }
        void updateMissRecordOnDevice()
        {
            CUDABuffer<MissRecord> d_miss_records;
            d_miss_records.copyToDevice(m_miss_records.data(), N * sizeof(MissRecord));

        }

        // ------------------------------------------------------------------
        // Hitgroup
        // ------------------------------------------------------------------
        void addHitgroupRecord(const std::array<HitgroupRecord, N>& hitgroup_records)
        {
            for (uint32_t i = 0; i < N; i++)
                m_hitgroup_records.emplace_back(hitgroup_records[i]);
        }

        /// @note 置き換えを行ったらデバイス上のデータも更新する？
        void replaceHitgroupRecord(const HitgroupRecord& record, const int idx) 
        {
            if (idx >= m_hitgroup_records.size())
            {
                pgLogFatal("The index out of range");
                return;
            }
            m_hitgroup_records[idx] = record;
        }

        void deleteHitgroupRecord(const int idx)
        {
            /// @todo Check if this works 
            if (m_hitgroup_records.empty()) 
                PG_LOG_WARN("An array of HitgroupRecords to be deleted is empty");
            if (m_hitgroup_records.size() <= idx)
                PG_LOG_WARN("The index", idx, "is out of bounds");
            m_hitgroup_records.erase(m_hitgroup_records.begin() + idx);
        }

        void updateHitgroupRecordOnDevice()
        {
            CUDABuffer<HitgroupRecord> d_hitgroup_records;
            d_hitgroup_records.copyToDevice(m_hitgroup_records);
            m_sbt.hitgroupRecordBase = d_hitgroup_records.devicePtr(); 
            m_sbt.hitgroupRecordCount = static_cast<uint32_t>(m_hitgroup_records.size());
            m_sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitgroupRecord));
        }

        uint32_t numHitgroupRecords() const 
        {
            return static_cast<uint32_t>(m_hitgroup_records.size());
        }

        HitgroupRecord& hitgroupRecord(const int idx) 
        {
            ASSERT(idx < (int)m_hitgroup_records.size(), "The index out of range.");
            return m_hitgroup_records[idx];
        }
        const HitgroupRecord& hitgroupRecord(const int idx) const 
        {
            ASSERT(idx < (int)m_hitgroup_records.size(), "The index out of range.");
            return m_hitgroup_records[idx];
        }
        CUdeviceptr deviceHitgroupRecordPtr() const
        {
            if (m_sbt.hitgroupRecordBase)
                return m_sbt.hitgroupRecordBase;
            else
            {
                THROW("Shader binding table hasn't been created on device yet.");
                return 0ull;
            }
        }

        // ------------------------------------------------------------------
        // Callables
        // ------------------------------------------------------------------
        template <class... CallablesRecordArgs>
        void addCallablesRecord(const CallablesRecordArgs&... args) {
            static_assert(std::conjunction<std::is_same<CallablesRecord, CallablesRecordArgs>...>::value, 
                "Record type must be same with 'CallablesRecord'.");

            pushToVector(m_callables_records, args...);
        }

        /// @note 置き換えを行ったらデバイス上のデータも更新する？
        void replaceCallablesRecord(const CallablesRecord& record, const int idx) {
            ASSERT(idx < (int)m_callables_records.size(), "The index out of range.");
            m_callables_records[idx] = record;
        }

        uint32_t numCallablesRecords() const 
        {
            return static_cast<uint32_t>(m_callables_records.size());
        }

        void deleteCallablesRecord(const int idx)
        {
            UNIMPLEMENTED();
        }

        CallablesRecord& callableRecord(const int idx)
        {
            ASSERT(idx < (int)m_callables_records.size(), "The index out of range.");
            return m_callables_records[idx];
        }
        const CallablesRecord& callableRecord(const int idx) const 
        {
            ASSERT(idx < (int)m_callables_records.size(), "The index out of range.");
            return m_callables_records[idx];
        }
        CUdeviceptr deviceCallablesRecordPtr() const 
        {
            if (m_sbt.callablesRecordBase)
                return m_sbt.callablesRecordBase;
            else
            {
                THROW("Shader binding table hasn't been created on device yet.");
                return 0ull;
            }
        }

        // ------------------------------------------------------------------
        // Exception
        // ------------------------------------------------------------------
        void setExceptionRecord(const ExceptionRecord& ex_record) {
            m_exception_record = ex_record;
        }
        ExceptionRecord& exceptionRecord()
        {
            return m_exception_record;
        }
        const ExceptionRecord& exceptionRecord() const
        {
            return m_exception_record;
        }
        CUdeviceptr deviceExceptionRecordPtr() const
        {
            if (m_sbt.exceptionRecord)
                return m_sbt.exceptionRecord;
            else
            {
                THROW("Shader binding table hasn't been created on device yet.");
                return 0ull;
            }
        }

        // ------------------------------------------------------------------
        // Utility
        // ------------------------------------------------------------------
        void createOnDevice() {
            d_raygen_record.copyToDevice(&m_raygen_record, sizeof(RaygenRecord));
            d_miss_records.copyToDevice(m_miss_records.data(), N * sizeof(MissRecord));
            d_hitgroup_records.copyToDevice(m_hitgroup_records);
            d_callables_records.copyToDevice(m_callables_records);
            d_exception_record.copyToDevice(&m_exception_record, sizeof(ExceptionRecord));

            m_sbt.raygenRecord = d_raygen_record.devicePtr();
            m_sbt.missRecordBase = d_miss_records.devicePtr();
            m_sbt.missRecordCount = N;
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

        void destroy() {
            if (m_sbt.raygenRecord) 
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
            if (m_sbt.missRecordBase) 
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
            if (m_sbt.hitgroupRecordBase) 
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
            if (m_sbt.callablesRecordBase)
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.callablesRecordBase)));
            if (m_sbt.exceptionRecord)
                CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.exceptionRecord)));
            m_sbt = {};
            m_raygen_record = {};
            m_miss_records = {};
            m_hitgroup_records.clear();
            m_callables_records.clear();
            m_exception_record = {};
            on_device = false;
        }

        OptixShaderBindingTable& sbt() 
        {
            return m_sbt;
        }

        bool isOnDevice() 
        {
            return on_device;
        }
    private:
        template <class Head, class... Args>
        static void pushToVector(std::vector<Head>& v, const Head& head, const Args&... args)
        {
            v.emplace_back(head);
            if constexpr (sizeof...(args) != 0)
                pushToVector(v, args...);
        }

        OptixShaderBindingTable m_sbt {};

        RaygenRecord                    m_raygen_record {};
        std::array<MissRecord, N>       m_miss_records;
        std::vector<HitgroupRecord>     m_hitgroup_records;
        std::vector<CallablesRecord>    m_callables_records;
        ExceptionRecord                 m_exception_record {};

        CUDABuffer<RaygenRecord>        d_raygen_record;
        CUDABuffer<MissRecord>          d_miss_records;
        CUDABuffer<HitgroupRecord>      d_hitgroup_records;
        CUDABuffer<CallablesRecord>     d_callables_records;
        CUDABuffer<ExceptionRecord>     d_exception_record;

        bool on_device;
    };

    // Default declaration for easy usage
    template <class Cam>
    using pgRaygenRecord = Record<pgRaygenData<Cam>>;
    using pgMissRecord = Record<pgMissData>;
    using pgHitgroupRecord = Record<pgHitgroupData>;
    using pgCallablesRecord = Record<pgEmptyData>;
    using pgExceptionRecord = Record<pgEmptyData>;

    template <class Cam, uint32_t N>
    using pgDefaultSBT = ShaderBindingTable<pgRaygenRecord<Cam>, pgMissRecord, pgHitgroupRecord, pgCallablesRecord, pgExceptionRecord, N>;

#endif // __CUDACC__

} // namespace prayground