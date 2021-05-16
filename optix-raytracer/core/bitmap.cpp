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
    m_data = new T[m_width*m_height];
}
template void Bitmap<uchar4>::allocate(int, int);
template void Bitmap<float4>::allocate(int, int);
template void Bitmap<uchar3>::allocate(int, int);
template void Bitmap<float3>::allocate(int, int);

// --------------------------------------------------------------------
/**
 * @todo T = float4 でも jpg ファイルを読み込みたい場合に対応する...?
 */
template <class T>
void Bitmap<T>::load(const std::string& filename) {
    // 画像ファイルが存在するかのチェック
    Assert(std::filesystem::exists(filename.c_str()), "The image file '" + filename + "' is not found.");

    using LoadableData = std::conditional_t< 
        (std::is_same_v<T, uchar4> || std::is_same_v<T, float4>), 
        uchar4, 
        uchar3
    >;

    std::string file_extension = get_extension(filename);

    if (file_extension == ".png" || file_extension == ".PNG")
        Assert(sizeof(loadable_type) == 4, "The type of bitmap must have 4 channels (RGBA) when loading PNG file.");
    else if (file_extension == ".jpg" || file_extension == ".JPG")
        Assert(sizeof(loadable_type) == 3, "The type of bitmap must have 3 channels (RGB) when loading JPG file.");

    LoadableData* data = reinterpret_cast<LoadableData*>(
        stbi_load(filename.c_str(), &m_width, &m_height, &m_channels, sizeof(loadable_type)));
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
                fd[idx] = color2float(d[idx]);
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
template <class T>
void Bitmap<T>::write(const std::string& filename, bool gamma_enabled, int quality) const
{
    // Tの方によって出力時の型をuchar4 or uchar3 で切り替える。
    using WritableData = std::conditional_t< 
        (std::is_same_v<T, uchar4> || std::is_same_v<T, float4>), 
        uchar4, 
        uchar3
    >;

    std::string file_extension = get_extension(filename);

    WritableData* data = new WritableData[m_width*m_height];
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
        stbi_write_png(filename, m_width, m_height, m_channels, data, 0);
    }
    else if (file_extension == ".jpg" || file_extension == ".JPG")
    {
        stbi_write_jpg(filename, m_width, m_height, m_channels, data, quality);
    }

    delete[] data;
}
template void Bitmap<float4>::write(const std::string&, bool, int) const;
template void Bitmap<uchar4>::write(const std::string&, bool, int) const;
template void Bitmap<float3>::write(const std::string&, bool, int) const;
template void Bitmap<uchar3>::write(const std::string&, bool, int) const;

// --------------------------------------------------------------------
template <class T>
void Bitmap<T>::copy_to_device() {
    // CPU側のデータが準備されているかチェック
    Assert(this->m_data, "Image data in the host side has been not allocated yet.");

    // GPU上に画像データを準備
    CUDABuffer<T> d_buffer;
    d_buffer.alloc_copy(this->m_data, this->m_width*this->m_height*sizeof(T));
    d_data = d_buffer.data();
}
template void Bitmap<uchar4>::copy_to_device();
template void Bitmap<uchar3>::copy_to_device();
template void Bitmap<float4>::copy_to_device();
template void Bitmap<float3>::copy_to_device();

}