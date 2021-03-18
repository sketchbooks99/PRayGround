# 開発用メモ

# CUDA
接頭語の意味
- 関数
  - `__global__` : ホスト側から呼び出されてデバイス側で実行される
  - `__device__` : デバイス側から呼び出されてデバイス側で実行される
  - `__host__` : ホスト側から呼び出されてホスト側で実行される
- 変数
  - `__device__` : グローバルメモリに格納される
    - `cudaMalloc()`によって領域確保される
    - 全スレッドからアクセス可能
  - `__shared__` : オンチップ共有メモリに格納される
  - 実行時 or コンパイル時に指定される 
    - 同じスレッドブロック内のすべてのスレッドでアクセス可能
    - 修飾子なし
    - スカラー型、ベクトル型はレジスタに格納される
    - レジスタに収まりきらないものはローカルメモリに溢れ出る。

## デバイス側にクラスのコピーオブジェクトを生成する。
`ref` : https://codereview.stackexchange.com/questions/193367/cuda-c-host-device-polymorphic-class-implementation

```c++
template <typename T, typename... Args>
__global__ void create_object_on_device(T** d_ptr, Args... args) {
  d_ptr = new T(args...);
}

class Material {
  Material* ptr();
protected:
  Material* d_ptr;
};

class Diffuse final : public Material {
public: 
  explicit Diffuse(const float3& albedo) : m_albedo(albedo) {}
  float3 albedo() const { return m_albedo; }
private:
  float3 m_albedo;
};

Diffuse* diffuse = new Diffuse(make_float3(0.8f));
Material* mat;
cudaMalloc(reinterpret_cast<void**>(&d_ptr), sizeof(diffuse));
create_object_on_device<<<1,1>>>(reinterpret_cast<Diffuse**>(&mat->ptr()), diffuse->albedo()); 
```

# OptiX

## プログラム
- Direct callable program
  - `optixTrace()` が呼べない
  - 関数から即座に呼べる。
  - `OptixDiractCallable()`で呼ぶ。
  - テクスチャ切り替えに向いている
- Continuation callable program
  - `optixTrace()`が呼べる。
  - スケジューラによって統制される必要があり、オーバーヘッドが大きくなりやすい。

## 関数
- `optixReportIntersection()` - OptiX Programing Guide p.72
  - 三角形は三角形内の重心座標(u,v)をattributeとして格納している。
    - 使用例) `float n = u*n0 + v*n1 + (1-u-v)*n2 // 法線の線形補間` 
```C++
__device__ bool optixReportIntersection(
  float hitT,                                  // ray time
  unsigned int hitKind,                        // specify any-hit or closest hit or both
  unsigned int a0, ..., unsigned int a7 );     // attributes (this is used to communicate with CH program)

__device__ unsigned int optixGetAttribute_0(); // 0 - 7
```

# CMake
- `ref` : https://theolizer.com/cpp-school-root/cpp-school3/
- `add_executable()`
  - 実行ファイルを生成するように指示

```cmake
add_executable(<name-of-exec> <src-files>)

ex)
add_executable(main main.cpp)
```

- `add_library()`
  - ビルドするライブラリ(.lib, .dll, .a, .so, .dylib, etc...)を定義する。
  - 使用するソースファイルやリンクするライブラリについては `add_executable` と同じ。

- `add_custom_target()`
  - ほとんどの場合、`add_executable`と`add_library`で足りるが、**Doxygen**などのコンパイラやりん海外のツールでファイル生成する場合に使うことが多い。

- `message(文字列)`

```cmake
message(WARNING "Be carefull!")
(output) CMake Warning at 01.cmake:1 (message): Be carefull!
message(STATUS, "Hello CMake!")
(output) -- Hello CMake!
```

- `set(変数名　値)`
  - 変数に値を設定

- マルチプラットフォーム対応

```cmake
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()
```

## ユーザー定義コマンド

```cmake
function(コマンド 仮引数リスト)
  # 処理
endfunction([コマンド名])

macro(コマンド名 仮引数リスト)
  # 処理
endmacro([コマンド名])

# ---- Example -----
# 定義
function(func0)
  message(STATUS "This is func0")
endfunction(func0)

# 呼び出し
func0()

# 引数有りマクロ
macro(macro1 ARG_A ARG_B)
  message(STATUS "macro1( '${ARG_A}' '${ARG_B}' )")
endmacro(macro1)
```

### 可変長パラメータ

| 変数名 | 意味 |
| - | - |
| ARGC | 実引数の数                           |
| ARGV | 実引数のリスト                       |
| ARGN | 仮引数に割当たらなかった実引数のリスト | 
| ARGV<番号> | argv[番号]                     | 

```cmake
function(func4 ARG)
  message(STATUS "ARGC=${ARGC}")
  
  foreach(ITEM ${ARGV})
    message(STATUS "ARGV=${ITEM}")
  endforeach()

  foreach(ITEM ${ARGV})
    message(STATUS "ARGV=${ITEM}")
  endforeach()

  message(STATUS "ARGV0=${ARGV0}")
endfunction(func4)

func4(param1 param2 param3)
```

### functionとmacroの違い
- `function`はC++の関数と似ている
  - 呼び出したところからfunctionに飛び、function文で定義された内容が実行される
- `macro`はC++の**プリプロセッサ・マクロ**のように振る舞う。
  - 定義した内容が、macroを呼び出したところへ**展開**されて実行される。

# Rule of coding 
## 変数・関数名
- クラス名は大文字で始める
- メンバ変数はできるだけ、`m_` で始める。
- プライベート関数は `_` で始める。
- クラス名以外は単語の間は `_` でつなぎ、大文字小文字の切り替えで単語を区切らないようにする
- `template` は使えるところでは積極的に使う。
  - 汎用性が高いクラス構造を目指す。
- 基本的に `using namespace` は避ける。名前空間をはっきり明記するように。**使う場合は局所的なスコープ(関数内、クラス内など)に限定する**
  - ex) 
  - NG: `using namespace std; string str;`;
  - OK: `std::string str;`

例
```c++
template <typename T>
class Hoge {
public:
    explicit Hoge(T val) : m_val(val), m_str("") {}
    explicit Hoge(T val, const std::string& str) : m_val(val), m_str(str) {}

    void set_val(T val) { m_val = val; }
    int get_val() const { return m_val; }

    void set_str(const std::string& str) { m_str = str; }
    std::string get_str() const { return m_str; }
private:
    T m_val;
    std::string m_str;
}
```