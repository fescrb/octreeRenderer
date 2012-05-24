#include "OpenCLRenderInfo.h"

cl_renderinfo convert(renderinfo info) {
	cl_int2 viewportSize = {info.viewportSize[0], info.viewportSize[1]};
	cl_float3 eyePos = {info.eyePos[0], info.eyePos[1], info.eyePos[2]};
	cl_float3 viewDir = {info.viewDir[0], info.viewDir[1], info.viewDir[2]};
	cl_float3 up = {info.up[0], info.up[1], info.up[2]};
	cl_float3 viewPortStart = {info.viewPortStart[0], info.viewPortStart[1], info.viewPortStart[2]};
	cl_float3 viewStep = {info.viewStep[0], info.viewStep[1], info.viewStep[2]};
	cl_renderinfo cl_info(viewportSize,
		eyePos,
		viewDir,
		up,
		viewPortStart,
		viewStep,
		info.eyePlaneDist,
		info.fov
	);
	
	return cl_info;
}