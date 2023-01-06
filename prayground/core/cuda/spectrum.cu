#include <prayground/core/spectrum.h>
#include <prayground/optix/macros.h>

namespace prayground {

    extern "C" HOST void initRGB2SpectrumTableOnGPU(
        SampledSpectrum& white, 
        SampledSpectrum& cyan, 
        SampledSpectrum& magenta, 
        SampledSpectrum& yellow, 
        SampledSpectrum& red, 
        SampledSpectrum& green,
        SampledSpectrum& blue)
    {
        cudaMemcpyToSymbol((void*)rgb2spectrum_white, &white, sizeof(SampledSpectrum));
        cudaMemcpyToSymbol((void*)rgb2spectrum_cyan, &cyan, sizeof(SampledSpectrum));
        cudaMemcpyToSymbol((void*)rgb2spectrum_magenta, &magenta, sizeof(SampledSpectrum));
        cudaMemcpyToSymbol((void*)rgb2spectrum_yellow, &yellow, sizeof(SampledSpectrum));
        cudaMemcpyToSymbol((void*)rgb2spectrum_red, &red, sizeof(SampledSpectrum));
        cudaMemcpyToSymbol((void*)rgb2spectrum_green, &green, sizeof(SampledSpectrum));
        cudaMemcpyToSymbol((void*)rgb2spectrum_blue, &blue, sizeof(SampledSpectrum));
    }

} // namespace prayground