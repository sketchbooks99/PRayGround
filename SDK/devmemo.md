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