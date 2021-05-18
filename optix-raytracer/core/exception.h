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

// ------------------------------------------------------------
class Exception : public std::runtime_error {

};

}