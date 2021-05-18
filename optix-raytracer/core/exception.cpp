#include "exception.h"

#include <cxxabi.h>
#include <typeinfo>
#include <format>

namespace oprt {

#ifdef _WIN32 || _WIN64

#include <Windows.h>
#include <ImageHlp.h>
#pragma comment(lib, "imagehlp.lib")

const StackTrace StackTracer::getStackTrace()
{
    // 最大トレースサイズ
    constexpr size_t max_size = 256;
    // トレースリスト
    void* traces[max_size] = {};

    // 現在のプロセスを取得
    HANDLE process = GetCurrentProcess();

    // シンボルハンドラの初期化
    SymInitialize(process, nullptr, TRUE);

    // スタックトレースの取得
    uint16_t trace_size = CaptureStackBackTrace(0, max_size, traces, nullptr);

    // シンボル名最大サイズをセット
    constexpr size_t max_name_size = 255;
    // シンボル情報サイズを算出
    constexpr size_t symbol_info_size = sizeof(SYMBOL_INFO) + ((max_name_size + 1) * sizeof(char));

    // シンボル情報のメモリ確保
    SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(symbol_info_size, 1);

    // スタックトレース情報生成
    StackTrace stack_trace;

    // シンボル情報メモリ確保成功
    if (symbol != nullptr)
    {
        // シンボル名最大サイズをセット
        symbol->MaxNameLen = max_name_size;
        // シンボル情報サイズをセット
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

        // トレースサイズをセット
        stack_trace.trace_size = static_cast<uint32_t>(trace_size);
        // トレースリストのメモリ確保
        stack_trace.traces.reserve(static_cast<size_t>(trace_size));
        // シンボルリストのメモリ確保
        stack_trace.symbols.reserve(static_cast<size_t>(trace_size));

        // トレースサイズ分ループ
        for (uint16_t i = 0; i < trace_size; i++)
        {
            // トレースアドレスからシンボル情報を取得
            SymFromAddr(process, (DWORD64)(traces[i]), 0, symbol);

            // トレースアドレスをトレースリストに追加
            stack_trace.traces.push_back(traces[i]);

            // シンボル名をシンボルリストに追加
            stack_trace.symbols.push_back(std::string(symbol->Name));
        }
        
        // シンボル情報のメモリ解放
        free(symbol);
    }
    else
    {

    }

    return stack_trace;
}

#elif __linux__

const StackTrace StackTracer::getStackTrace()
{
    // 最大トレースサイズ
    constexpr size_t max_size = 256;
    // トレースリスト
    void* traces[max_size] = {};

    // スタックトレース取得
    int trace_size = backtrace(traces, max_size);
    // シンボルリスト取得
    char** symbols = backtrace_symbols(traces, trace_size);

    // スタックトレース情報生成
    StackTrace stack_trace;

    // トレースサイズ
    stack_trace.trace_size = static_cast<uint32_t>(trace_size);

    // トレースリストメモリ確保
    stack_trace.traces.reserve(trace_size);
    // シンボルリストメモリ確保
    stack_trace.symbols.reserve(trace_size);

    // トレースサイズ分ループ
    for (int i = 0; i < trace_size; i++)
    {
        // トレースアドレスをリストに追加
        stack_trace.traces.push_back(traces[i]);

        // シンボル情報をシンボルリストに追加
        stack_trace.symbols.push_back(symbols[i]);
    }

    // シンボルリスト解放
    free(symbols);

    return stack_trace;
}

#else

const StackTrace StackTracer::getStackTrace()
{
    // 空のスタックトレース情報生成
    StackTrace stack_trace;
    stack_trace.trace_size = 0;
    stack_trace.traces.clear();
    stack_trace.symbols.clear();

    return stack_trace;
}

#endif

SymbolInfo getSymbolInfo(const std::string& raw_symbol_info)
{
    // シンボル情報パターン : <ObjectName>(<MangledSymbolName>+<Offset>) [<Address>]
    std::regex pattern("^(.+)\\((\\w*)\\+(0x[0-9a-fA-F]+)\\)\\s+\\[(0x[0-9a-fA-F]+)\\]$");
    std::smatch sm;

    // シンボル情報生成
    SymbolInfo symbol_info;

    // シンボル情報パターンにマッチ
    if (std::regex_match(raw_symbol_info, sm, pattern))
    {
        // 有効なシンボル情報生成
        symbol_info.object_name = sm[1].str();
        symbol_info.mangled_symbol_name = sm[2].str();
        symbol_info.offset = sm[3].str();
        symbol_info.address = sm[4].str();
        symbol_info.is_valid = true;
    }
    else 
    {
        // 無効なシンボル情報生成
        symbol_info.object_name = "";
        symbol_info.address = "";
        symbol_info.mangled_symbol_name = "";
        symbol_info.offset = "";
        symbol_info.is_valid = false;
    }

    return symbol_info;
}

std::string getSymbolInfoText(const std::string& raw_symbol_info)
{
    // シンボル情報を取得
    const SymbolInfo symbol_info = getSymbolInfo(raw_symbol_info);

    // シンボル情報テキストを初期化
    std::string symbol_info_text = "";

    if (symbol_info.is_valid == true)
    {
        // マングルされたシンボル名を取得
        std::string mangled_symbol_name = symbol_info.mangled_symbol_name;

        // マングルされたシンボル名を判定
        if (mangled_symbol_name[0] == '_')
        {
            // デマングルされたシンボル名を取得
            int status = 0;
            char* demangled_symbol_name = abi::_cxa_demangle(mangled_symbol_name.c_str(), 0, 0, &status);

            // デマングルに成功したらデマングルされたシンボル名を関数名にセット、失敗したらマングルされたシンボル名を使用
            std::string function_name = (status == 0) ? demangled_symbol_name : mangled_symbol_name;

            // デマングルされたシンボル名のメモリ解放
            free(demangled_symbol_name);

            // シンボル情報テキストをセット
            symbol_info_text = std::format("{} {} + {}", symbol_info.object_name, function_name, symbol_info.offset);
        }
        else
        {
            // シンボル情報テキストをセット
            symbol_info_text = std::format("{} {} + {}", symbol_info.object_name, mangled_symbol_name, symbol_info.offset);
        }
    }
    else
    {
        // シンボル情報テキストをセット
        symbol_info_text = raw_symbol_info;
    }

    return symbol_info_text;
}

}