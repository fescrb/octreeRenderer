#include "OpenCLRenderInfo.h"

cl_renderinfo convert(renderinfo info) {
	cl_renderinfo cl_info;
	
	// cl_int
	cl_info.maxOctreeDepth = info.maxOctreeDepth;
	
	// cl_int2
	cl_int2 viewportSize = {info.viewportSize[0], info.viewportSize[1]};
	cl_info.viewportSize = viewportSize;

	//cl_int3
	/*cl_int3 eyePos = {info.eyePos[0], info.eyePos[1], info.eyePos[2]};
	cl_info.eyePos = eyePos;
	cl_int3 viewDir = {info.viewDir[0], info.viewDir[1], info.viewDir[2]};
	cl_info.viewDir = viewDir;
	cl_int3 up = {info.up[0], info.up[1], info.up[2]};
	cl_info.up = up;
	cl_int3 viewPortStart = {info.viewPortStart[0], info.viewPortStart[1], info.viewPortStart[2]};
	cl_info.viewPortStart = viewPortStart;
	cl_int3 viewStep = {info.viewStep[0], info.viewStep[1], info.viewStep[2]};
	cl_info.viewStep = viewStep;*/
	
	//cl_float
	cl_info.eyePlaneDist = info.eyePlaneDist;
	cl_info.fov = info.fov;
	
	return cl_info;
}