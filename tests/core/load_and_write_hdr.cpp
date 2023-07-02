#include <prayground/prayground.h>

using namespace std;

int main()
{
    string infile = "resources/image/sepulchral_chapel_basement_4k.exr";
    auto hdr_out = pgPathJoin(pgRootDir(), "resources/image/debug.hdr");
    auto exr_out = pgPathJoin(pgRootDir(), "resources/image/debug.exr");
    auto png_out = pgPathJoin(pgRootDir(), "resources/image/debug.png");

    FloatBitmap fbmp(infile);
    fbmp.write(hdr_out);
    fbmp.write(exr_out);
    fbmp.write(png_out);

    const int32_t w = 512, h = 512;
    FloatBitmap fbmp2(PixelFormat::RGBA, w, h);
    float4* data = new float4[w * h];
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            data[y * w + x] = make_float4((float)x / w, (float)y / w, 0.0f, (float)(x * y) / (w * h));
        }
    }
    fbmp2.setData(reinterpret_cast<float*>(data), 0, 0, w, h);
    auto hdr_out2 = pgPathJoin(pgRootDir(), "resources/image/debug2.hdr");
    auto exr_out2 = pgPathJoin(pgRootDir(), "resources/image/debug2.exr");
    auto png_out2 = pgPathJoin(pgRootDir(), "resources/image/debug2.png");
    fbmp2.write(hdr_out2);
    fbmp2.write(exr_out2);
    fbmp2.write(png_out2);
    return 0;
}