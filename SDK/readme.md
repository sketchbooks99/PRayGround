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