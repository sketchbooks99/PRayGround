#
#  Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

if( CUDA_VERSION VERSION_LESS "11.1" )
  message( WARNING "Skipping DemandLoading library and samples, which require CUDA 11.1 or later." )
  return()
endif()

set( target_name DemandLoading_exp )

add_library( ${target_name} STATIC
  CheckerBoardImage.cpp
  include/DemandLoading/CheckerBoardImage.h
  include/DemandLoading/DemandTexture.h
  DemandTextureImpl.cpp
  DemandTextureImpl.h
  include/DemandLoading/DemandTextureContext.h
  include/DemandLoading/DemandTextureInfo.h
  include/DemandLoading/DemandTextureManager.h
  DemandTextureManagerImpl.cpp
  DemandTextureManagerImpl.h
  Exception.h
  ExtensibleArray.h
  ImageReader.cpp
  include/DemandLoading/ImageReader.h
  Math.h
  PageTableManager.h
  SparseTexture.cpp
  SparseTexture.h
  include/DemandLoading/Tex2D.h
  include/DemandLoading/TextureDescriptor.h
  TextureInfo.cpp
  include/DemandLoading/TextureInfo.h
  include/DemandLoading/TileIndexing.h
  TilePool.cpp
  TilePool.h
  # EXRReader.cpp is added below if OpenEXR is available.
  )

source_group( "Header Files\\Implementation" FILES
  DemandTextureImpl.h
  DemandTextureManagerImpl.h
  Exception.h
  ExtensibleArray.h
  PageTableManager.h
  SparseTexture.h
  TilePool.h
  )

# Not sure why glad include directory must be specified here.  It should come with the library dependency below,
# but that doesn't seem to be working on the Mac.
target_include_directories( ${target_name} PUBLIC
  include
  ../../support
  )

target_link_libraries( ${target_name} PUBLIC
  optixPaging
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )

find_package( OpenEXR )
if( NOT OpenEXR_FOUND )
  message( WARNING "OpenEXR not found (see OpenEXR_ROOT).  Will use procedural texture in optixDemandTextureAdvanced." )
endif()

find_package( ZlibStatic )
if ( NOT ZlibStatic_FOUND )
  message( WARNING "Zlib not found (see ZlibStatic_ROOT).  Will use procedural texture in optixDemandTextureAdvanced." )
endif()

if( OpenEXR_FOUND AND ZlibStatic_FOUND )
  # Build our EXRImage class if OpenEXR is available, adding the necessary include path and libraries.
  target_sources( ${target_name} PRIVATE
    EXRReader.cpp
    include/DemandLoading/EXRReader.h )
  target_compile_definitions( ${target_name} PUBLIC OPTIX_SAMPLE_USE_OPEN_EXR )
  target_link_libraries( ${target_name} PUBLIC OpenEXR::IlmImf Zlib::Static )
endif()

set_property(TARGET ${target_name} PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")
