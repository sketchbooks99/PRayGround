:warning: このリポジトリはまだ開発途中で、一部に破壊的な変更が加わる可能性があります。

Languages | [English](README.md) | 日本語

# OptiX-Raytracer

OptiX 7 をベースとしたレイトレーサーです。OptiXのAPIを意識することなく、基本的にはシーンの記述（座標変換やジオメトリ、マテリアル等）のみでレンダリングが可能です。
さらに、OptiX 7 の煩雑な処理に対するラッパーライブラリと、ユーザー定義によるジオメトリやマテリアル、テクスチャの簡易な登録システムを提供します。

![output.png](result/016_env.jpg)

# :computer: Requirements
プロジェクトをビルドする前に、以下の要件、特にC++のコンパイラが `<concepts>` ヘッダー等のC++20の機能をサポートしているかどうか確認してください。

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
コンパイルする前に，環境変数 `CC` と `CXX` をエクスポートしてください。これらの環境変数は、ターミナルを起動したときに自動的にエクスポートされるように、`~/.bashrc` ファイルへの記述を推奨します。
```
export CC=gcc-10
export CXX=g++-10
```
次に、以下のコマンドでビルドします。ターミナルで`ccmake`を実行するときもしくはccmakeのプロンプトで必ず`OptiX_INCLUDE`を設定してください。
```
cd <path/to/OptiX-Raytracer>
mkdir build
cd build

ccmake .. -DOptiX_INCLUDE=<path/to/OptiX>/include
or 
ccmake .. # and set OptiX_INCLUDE to the path of OptiX library include.
# You can generate a makefile with a standard step of CMake ( [C] Configure -> [G] Generate ).

make
```

コンパイルが完了すると、`build/bin`ディレクトリに実行ファイルが生成されます。
```
cd bin
./oprt 
```

## Windows
Windowsでは、C++20の機能をサポートしている**Visual Studio 2019**の最新版が必要です。

CMakeの実行では、[cmake-gui](https://cmake.org/download/)を使用してください。

ビルドの手順は以下のとおりです。

1. cmake-gui を起動する。

2. ソースコードの場所（**Where is the source code**）には、`<path/to/OptiX-Raytracer>` を設定してください。

3. バイナリの場所（**Where to build the binaries**）には、`<path/to/OptiX-Raytracer>/build` を設定してください。

4. ウィンドウの下部にある `Configure` ボタンを押してください。ポップアップウィンドウが表示されたら、注意してビルドするプラットフォームの設定を選択してください。最近のC++の機能を使うためには、ジェネレーターとして **Visual Studio 16 2019** を選択し、OptiXは64-bitのビルドしかサポートしていないため、ジェネレーターのプラットフォームには **x64** を指定してください。

5. `Finish` ボタンを押すとビルド情報の収集が始まります。処理中にエラーが発生する場合は、`OptiX_INCLUDE_NOTFOUND` に起因するエラーの可能性があります。その場合は、`OptiX_INCLUDE` にOptiXのインクルードディレクトリのパスを設定してください。Windowsの場合、OptiXのインクルードディレクトリは、`C:ProgramData\NVIDIA Corporation\OptiX SDK <your version>` にあります。

6. 処理が完了したら, `Generate` ボタンを押してください。

7. `build/` ディレクトリにある `OptiX-Raytracer.sln` というソリューションファイルを開いてください。

8. IDEで `ソリューションのビルド` を実行します。コンパイルが成功すると、実行ファイルが `build/bin/Debug or Release` に作成されます。また、`oprt` プロジェクトをスタートアッププロジェクトとして設定し、`ローカル Windows デバッガー` を実行することで、レイトレーサーを簡単に実行できます。
    - :warning: Japanese comments included in our sources may cause errors during compliation due to the encoding. So, please add /source-charset:utf-8 for Additional Options of Command Line. (Project tab -> Properties -> C/C++ -> Command Line) 
    - :warning: ソースに含まれる日本語コメントのエンコーディングによってエラーが発生しコンパイルが通らない場合があります。その場合には /source-charset:utf-8 オプションを Additional Options of Command Line に追加してください。 (Project tab -> Properties -> C/C++ -> Command Line) 

## Mac
サポートしていません。

# Modyfying the scene

現在のところ、マテリアル、ジオメトリ、テクスチャ、センサーなどのシーン・パラメータを変更するには、`scene_config.h` を修正するか、シーンを記述して作成したヘッダファイルを `oprt.cpp` でインクルードしてください。

将来的には、pbrtやmitsuba2で実装されているような、シーン構成のランタイムロードを実装したいと考えています。また、シーンをxmlファイルなどの別ファイルから読み込むか、APIを使って直接コードに書き込むかを選択できるようにしたいと考えています。

