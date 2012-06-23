#include "CUDARenderInfo.h"

cuda_render_info cudaConvert(renderinfo *info) {
    cuda_render_info cuda_info;
    
    cuda_info.eyePos = make_float3(info->eyePos.getX(),info->eyePos.getY(),info->eyePos.getZ());
    cuda_info.viewDir = make_float3(info->viewDir.getX(),info->viewDir.getY(),info->viewDir.getZ());
    cuda_info.up = make_float3(info->up.getX(),info->up.getY(),info->up.getZ());
    cuda_info.viewPortStart = make_float3(info->viewPortStart.getX(),info->viewPortStart.getY(),info->viewPortStart.getZ());
    cuda_info.viewStep = make_float3(info->viewStep.getX(),info->viewStep.getY(),info->viewStep.getZ());
    
    cuda_info.eyePlaneDist = info->eyePlaneDist;
    cuda_info.fov = info->fov;
    cuda_info.pixel_half_size = info->pixel_half_size;
    
    cuda_info.lightPos = make_float3(info->lightPos.getX(),info->lightPos.getY(),info->lightPos.getZ());
    
    cuda_info.lightBrightness = info->lightBrightness;
    
    return cuda_info;
}