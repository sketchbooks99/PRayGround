# 開発用メモ

## cudaのマクロとか

```c++
// At cuda
#if defined(__CUDACC__) || defined(__CUDABE__)
#    define SUTIL_HOSTDEVICE __host__ __device__
#    define SUTIL_INLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
// At cpp
#else 
#    define SUTIL_HOSTDEVICE
#    define SUTIL_INLINE inline
#    define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif
```

# Rule of coding 
## 変数・関数名
- クラス名は大文字で始める
- メンバ変数はできるだけ、`m_` で始める。
- プライベート関数は `_` で始める。
- クラス名以外は単語の間は `_` でつなぎ、大文字小文字の切り替えで単語を区切らないようにする
- `template` は使えるところでは積極的に使う。
    - 汎用性が高いクラス構造を目指す。
- 基本的に `using namespace` は避ける。名前空間をはっきり明記するように。**使う場合は局所的なスコープ(関数内、クラス内など)**
    - ex) 
        - NG: `using namespace std; string str`;
        - OK: `std::string str;`


例
```c++
class Hoge {
public:
    explicit Hoge(int val) : m_val(val), m_str("") {}
    explicit Hoge(int val, const std::string& str) : m_val(val), m_str(str) {}

    void set_val( int val ) { m_val = val; }
    int get_val() const { return m_val; }

    void set_str( const std::string& str) { m_str = str; }
    std::string get_str() const { return m_str; }
private:
    void _init() { m_val = 0; m_str = ""; }

    int m_val;
    std::string m_str;
}
```