# OptiX
このリポジトリはNVIDIA OptiX Raytracing Engine にまつわる開発を進めるリポジトリです。

This repository is for developping a ray tracing application with NVIDIA OptiX Raytracing Engine.

# TODO
- [x] .objのローダーをblender由来のオブジェクトも読み込めるように改変する。
- [ ] Transform( translate/rotate/scale )を適用できるようにする。
- [ ] 下のコードみたいにMaterial と Shape オブジェクトを一緒に管理する？
```c++
struct Primitive {
    Shape or Mesh shape; // Is it better to manage with the pointer of abstract class?
    Material material;
};

// before
std::vector<TriangleMesh> meshes;
std::vector<Material*> materials;
meshes.emplace_back(mesh);
materials.emplace_back(new Material());

// after
std::vector<Primitive> primitives;
primitives.emplace_back(mesh, new Material());
```