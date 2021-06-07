/**
 * @file exception.h
 * @author Shunji Kiuchi 
 * @brief Management of stack trace and exception.
 * 
 * @ref https://an-embedded-engineer.hateblo.jp/entry/2020/08/24/212511
 */

#pragma once

#include "util.h"

namespace oprt {

// ------------------------------------------------------------
struct SymbolInfo
{
    SymbolInfo()
    : is_valid(false), object_name(""), address(""),
      mangled_symbol_name(""), offset("") { }

    // シンボル情報有効のフラグ
    bool is_valid;
    // オブジェクト名
    std::string object_name;
    // アドレス
    std::string address;
    // マングルされたシンボル名
    std::string mangled_symbol_name;
    // オフセット
    std::string offset;
};

SymbolInfo getSymbolInfo(const std::string& raw_symbol_info);
std::string getSymbolInfoText(const std::string& raw_symbol_info);

// ------------------------------------------------------------
struct StackTrace {
    // トレース数
    uint32_t trace_size;
    // トレースアドレスリスト
    std::vector<void*> traces;
    // トレースシンボルリスト
    std::vector<std::string> symbols;
};

// ------------------------------------------------------------
class StackTracer {
public:
    static const StackTrace getStackTrace();
};

}

#ifndef __CUDACC__
#define OPTIX_CHECK_TRACE( call )                                                                           \
    do                                                                                                      \
    {                                                                                                       \
        OptixResult res = call;                                                                             \
        if( res != OPTIX_SUCCESS )                                                                          \
        {                                                                                                   \
            oprt::StackTrace stack_trace = oprt::StackTracer::getStackTrace();                              \
            std::stringstream ss;                                                                           \
            ss << "[STACK TRACE] " << std::endl;                                                            \
            for (size_t i = 0; i < stack_trace.trace_size; i++)                                             \
            {                                                                                               \
                if (i != 0)                                                                                 \
                {                                                                                           \
                    ss << std::endl;                                                                        \
                }                                                                                           \
                ss << " ";                                                                                  \
                ss << std::setw(16) << std::setfill('0') << std::hex << (uint64_t)stack_trace.traces[i];    \
                ss << " | ";                                                                                \
                ss << stack_trace.symbols[i];                                                               \
            }                                                                                               \
            ss << std::endl;                                                                                \
            std::cout << ss.str();                                                                          \
            ss.str("");                                                                                     \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"                                      \
               << __LINE__ << ")\n";                                                                        \
            throw sutil::Exception( res, ss.str().c_str() );                                                \
        }                                                                                                   \
    } while( 0 )
#endif