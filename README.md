# OptiX-Raytracer

OptiX 7 をベースとしたレイトレーサーです。OptiXのAPIを意識することなく、基本的にはシーンの記述（座標変換やジオメトリ、マテリアル等）のみでレンダリングが可能です。
さらに、OptiX 7 の煩雑な処理に対するラッパーライブラリと、ユーザー定義によるジオメトリやマテリアル、テクスチャの簡易な登録システムを提供します。


This is a ray tracer based on OptiX 7. Basically, this allows you to render  just by describing the scene (transformations, geometry, materials, etc.) without being aware of the OptiX API. This also provides a wrapper library for OptiX 7 and a simple registration system for user-defined geometries, materials and textures.

![output.png](result/output.png)