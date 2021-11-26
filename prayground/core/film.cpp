#include "film.h"

namespace prayground {

// ------------------------------------------------------------------
Film::Film()
{
    Message(MSG_WARNING, "This class is deprecated because it doesn't have much usage, and will be deleted soon.");
}

// ------------------------------------------------------------------
Film::Film(int width, int height)
: m_width(width), m_height(height)
{
    Message(MSG_WARNING, "This class is deprecated because it doesn't have much usage, and will be deleted soon.");
}

Film::~Film()
{

}

// ------------------------------------------------------------------
void Film::addBitmap(const std::string& name, PixelFormat format)
{
    if (format == PixelFormat::NONE)
    {
        Message(MSG_FATAL, "prayground::Film::addBitmap(): The unknown format.");
        return;
    }

    if (m_width == 0 || m_height == 0)
    {
        Message(MSG_FATAL, "prayground::Film::addBitmap(): The resolution of Film is invalid value.");
        return;
    }

    m_bitmaps.emplace(name, std::make_shared<Bitmap>(format, m_width, m_height));
}

std::shared_ptr<Bitmap> Film::bitmapAt(const std::string& name) const
{
    auto it = m_bitmaps.find(name);
    ASSERT(it != m_bitmaps.end(), "A bitmap layer associated with the key name of '" + name + "' is not found.");
    return it->second;
}

std::vector<std::shared_ptr<Bitmap>> Film::bitmaps() const 
{
    std::vector<std::shared_ptr<Bitmap>> tmp(m_bitmaps.size()); 
    std::transform(m_bitmaps.begin(), m_bitmaps.end(), tmp.begin(), [](auto pair) { return pair.second; });
    return tmp;
}

size_t Film::numBitmaps() const
{
    return m_bitmaps.size();
}

// ------------------------------------------------------------------
void Film::addFloatBitmap(const std::string& name, PixelFormat format)
{
    if (format == PixelFormat::NONE)
    {
        Message(MSG_FATAL, "prayground::Film::addFloatBitmap(): The unknown format.");
        return;
    }

    if (m_width == 0 || m_height == 0)
    {
        Message(MSG_FATAL, "prayground::Film::addFloatBitmap(): The resolution of Film is invalid value.");
        return;
    }

    m_float_bitmaps.emplace(name, std::make_shared<FloatBitmap>(format, m_width, m_height));
}

std::shared_ptr<FloatBitmap> Film::floatBitmapAt(const std::string& name)
{
    auto it = m_float_bitmaps.find(name);
    ASSERT(it != m_float_bitmaps.end(), "A bitmap layer associated with the key name of '" + name + "' is not found.");
    return it->second;
}

std::vector<std::shared_ptr<FloatBitmap>> Film::floatBitmaps() const 
{
    std::vector<std::shared_ptr<FloatBitmap>> tmp(m_float_bitmaps.size()); 
    std::transform(m_float_bitmaps.begin(), m_float_bitmaps.end(), tmp.begin(), [](auto pair) { return pair.second; });
    return tmp;
}

size_t Film::numFloatBitmaps() const 
{
    return m_float_bitmaps.size();
}

// ------------------------------------------------------------------
void Film::setResolution(int32_t width, int32_t height)
{
    m_width = width; m_height = height;
}

void Film::setWidth(int32_t width)
{
    m_width = width;
}

void Film::setHeight(int32_t height)
{
    m_height = height;
}

int32_t Film::width() const 
{
    return m_width;
}

int32_t Film::height() const 
{
    return m_height;
}

} // ::prayground