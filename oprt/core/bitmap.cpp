#include "bitmap.h"
#include "color.h"
#include "cudabuffer.h"
#include "file_util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../ext/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ext/stb/stb_image_write.h"

namespace oprt {

// --------------------------------------------------------------------
template <class T>
void Bitmap<T>::allocate(int width, int height)
{
    m_width = width; m_height = height;
    Assert(!this->m_data, "Image data in the host side is already allocated.");

    using Element_t = std::conditional_t<
        std::is_same_v<T, float4> || std::is_same_v<T, float3>, 
        float, 
        unsigned char
    >;
    constexpr int num_element = static_cast<int>(sizeof(T) / sizeof(Element_t));
    std::vector<Element_t> zero_arr(num_element * m_width * m_height, static_cast<Element_t>(0));

    m_data = new T[m_width*m_height];
    memcpy(m_data, zero_arr.data(), sizeof(T) * m_width * m_height);
}
template void Bitmap<uchar4>::allocate(int, int);
template void Bitmap<float4>::allocate(int, int);
template void Bitmap<uchar3>::allocate(int, int);
template void Bitmap<float3>::allocate(int, int);

// --------------------------------------------------------------------
/**
 * @todo 
 * - T = float4 でも jpg ファイルを読み込みたい場合に対応する...?
 * - テンプレートの特殊化
 */
template <class T>
void Bitmap<T>::load(const std::string& filename) {
    // 画像ファイルが存在するかのチェック
    Assert(std::filesystem::exists(filename.c_str()), "The image file '" + filename + "' is not found.");

    using Loadable_t = std::conditional_t< 
        (std::is_same_v<T, uchar4> || std::is_same_v<T, float4>), 
        uchar4, 
        uchar3
    >;

    std::string file_extension = getExtension(filename);

    if (file_extension == ".png" || file_extension == ".PNG")
        Assert(sizeof(Loadable_t) == 4, "The type of bitmap must have 4 channels (RGBA) when loading PNG file.");
    else if (file_extension == ".jpg" || file_extension == ".JPG")
        Assert(sizeof(Loadable_t) == 3, "The type of bitmap must have 3 channels (RGB) when loading JPG file.");

    Loadable_t* data = reinterpret_cast<Loadable_t*>(
        stbi_load(filename.c_str(), &m_width, &m_height, &m_channels, sizeof(Loadable_t)));
    Assert(data, "Failed to load image file'" + filename + "'");

    m_data = new T[m_width * m_height];
    // Tの型がfloatの場合は中身を unsigned char [0, 255] -> float [0.0f, 1.0f] に変換する
    if constexpr (std::is_same_v<T, float4> || std::is_same_v<T, float3>)
    {
        T* float_data = new T[m_width*m_height];
        for (auto i=0; i<m_width; i++)
        {
            for (auto j=0; j<m_height; j++)
            {
                auto idx = i * m_height + j;
                float_data[idx] = color2float(data[idx]);
            }
        }
        memcpy(m_data, float_data, m_width*m_height*sizeof(T));
        delete[] float_data;
    }
    else
    {
        memcpy(m_data, data, m_width*m_height*sizeof(T));
    }
    stbi_image_free(data);
}
template void Bitmap<uchar4>::load(const std::string&);
template void Bitmap<float4>::load(const std::string&);
template void Bitmap<uchar3>::load(const std::string&);
template void Bitmap<float3>::load(const std::string&);

// --------------------------------------------------------------------
/**
 * @todo
 * テンプレートの特殊化
 */
template <class T>
void Bitmap<T>::write(const std::string& filename, bool gamma_enabled, int quality) const
{
    // Tの方によって出力時の型をuchar4 or uchar3 で切り替える。
    using Writable_t = std::conditional_t< 
        (std::is_same_v<T, uchar4> || std::is_same_v<T, float4>), 
        uchar4, 
        uchar3
    >;

    std::string file_extension = getExtension(filename);

    Writable_t* data = new Writable_t[m_width*m_height];
    // Tの型がfloatの場合は中身を float [0.0f, 1.0f] -> unsigned char [0, 255] に変換する
    if constexpr (std::is_same_v<T, float4> || std::is_same_v<T, float3>)
    {
        for (int i=0; i<m_width; i++)
        {
            for (int j=0; j<m_height; j++)
            {
                int idx = i * m_height + j;
                data[idx] = make_color(m_data[idx], gamma_enabled);
            }
        }
    }
    else 
    {
        memcpy(data, m_data, m_width*m_height*sizeof(T));
    }
    
    if (file_extension == ".png" || file_extension == ".PNG")
    {
        stbi_flip_vertically_on_write(true);
        stbi_write_png(filename.c_str(), m_width, m_height, m_channels, data, 0);
    }
    else if (file_extension == ".jpg" || file_extension == ".JPG")
    {
        stbi_write_jpg(filename.c_str(), m_width, m_height, m_channels, data, quality);
    }

    delete[] data;
}
template void Bitmap<float4>::write(const std::string&, bool, int) const;
template void Bitmap<uchar4>::write(const std::string&, bool, int) const;
template void Bitmap<float3>::write(const std::string&, bool, int) const;
template void Bitmap<uchar3>::write(const std::string&, bool, int) const;

// --------------------------------------------------------------------
template <class T>
void Bitmap<T>::copyToDevice() {
    // CPU側のデータが準備されているかチェック
    Assert(m_data, "Image data in the host side has been not allocated yet.");

    // GPU上に画像データを準備
    CUDABuffer<T> d_buffer;
    d_buffer.copyToDevice(m_data, m_width*m_height*sizeof(T));
    d_data = d_buffer.deviceData();
}
template void Bitmap<uchar4>::copyToDevice();
template void Bitmap<uchar3>::copyToDevice();
template void Bitmap<float4>::copyToDevice();
template void Bitmap<float3>::copyToDevice();

}