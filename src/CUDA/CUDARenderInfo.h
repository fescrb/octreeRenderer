#ifndef _CUDA_RENDER_INFO_H
#define _CUDA_RENDER_INFO_H

#include "CUDAIncludes.h"

#include "RenderInfo.h"

struct cuda_render_info {
    float3 eyePos, viewDir, up, viewPortStart, viewStep;
    float eyePlaneDist, fov, pixel_half_size;
    
    float3 lightPos;
    float lightBrightness;
};

cuda_render_info cudaConvert(renderinfo *info);

#endif //_CUDA_RENDER_INFO_H