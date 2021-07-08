:warning: This repository is still under development, and parts of the project are subject to destructive changes.

# OptiX-Raytracer

OptiX 7 をベースとしたレイトレーサーです。OptiXのAPIを意識することなく、基本的にはシーンの記述（座標変換やジオメトリ、マテリアル等）のみでレンダリングが可能です。
さらに、OptiX 7 の煩雑な処理に対するラッパーライブラリと、ユーザー定義によるジオメトリやマテリアル、テクスチャの簡易な登録システムを提供します。


This is a ray tracer based on OptiX 7. Basically, this allows you to render just by describing the scene (transformations, geometry, materials, etc.) without being aware of the OptiX API. This also provides a wrapper library for OptiX 7 and a simple registration system for user-defined geometries, materials and textures.

![output.png](result/016_env.jpg)

# :computer: Requirements
Before building our codes, please be sure to check requirements and your environment, especially please make sure that your version of the C++ compiler supports C++20 features such as `<concepts>` header of the C++ standard libraries.

- CUDA Toolkit (Tested : 11.1, 11.2)
- C++ compiler which supports C++20 
    - Linux (Tested : g++ 10.3.0)
    - Windows (Tested : Visual Studio 2019, version 16.10.2) 
- OptiX 7 (Tested : 7.1, 7.2)
- CMake 3.0 minimum (Tested : cmake 3.16.3)

# :inbox_tray: Cloning
```
git clone https://github.com/uec-media-design-lab/OptiX-Raytracer.git
cd OptiX-Raytracer
git submodule update --init --recursive 
```

# :hammer: How to build
## Linux
---
Before compiling, please be sure to export two environment variables `CC` and `CXX`. We recommend you to add them to your `~/.bashrc` file as follows so that they are automatically exported when the terminal are launched.
```
export CC=gcc-10
export CXX=g++-10
```
Next, you can build with following command. Please be sure to set `OptiX_INCLUDE` in terminal execution of ccmake or in ccmake setting.
```
cd <path/to/OptiX-Raytracer>
mkdir build
cd build

ccmake .. -DOptiX_INCLUDE=<path/to/OptiX>/include
or 
ccmake .. # and set OptiX_INCLUDE to the path of OptiX library include.

make
```

After compiling got through, the execution file will be created in `build/bin` directory. 
```
cd bin
./oprt 
```

## Windows
---
On Windows, a recent version of **Visual Studio 2019** which supports C++20 features is required.

For configuring sources, please use cmake-gui to generate the solution files.

Building steps are as follows.

1. Start up cmake-gui.

2. Set the `<path/to/OptiX-Raytracer>` to the source code location ( **Where is the source code** ). 

3. Set the `<path/to/OptiX-Raytracer>/build` to the binary location ( **Where to build the binaries** ).

4. Press `Configure` button at the bottom of the window. When a window popped up, be careful with the settings for the platform to build. You must select **Visual Studio 16 2019** as the generator to use the recent features of C++, and specify the **x64** for the generator platform because OptiX only supports 64-bit builds.

5. Press `Finish` button and configuration will start. If errors occur while configuring, it may be due to `OptiX_INCLUDE_NOTFOUND` error. If so, please set `OptiX_INCLUDE` to the path of Optix library include. On Windows, OptiX include directory may be located in `C:ProgramData\NVIDIA Corporation\OptiX SDK <your version>`.

6. When configulation finished, press `Generate`.

7. Open the `OptiX-Raytracer.sln` solution file in the build directory.

8. Execute `Build Solution` in the IDE. When compile succeeded, executive file will be created in `build/bin/Debug or Release`. You also can run the ray tracer by setting the `oprt` project as start up project.

## Mac
---
Not supported.