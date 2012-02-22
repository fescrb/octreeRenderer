#ifndef _OPENCL_RENDER_INFO_H
#define _OPENCL_RENDER_INFO_H

#ifdef _LINUX
    #include <CL/cl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

#include "RenderInfo.h"

struct cl_renderinfo {
	cl_int   maxOctreeDepth;
    
    cl_int2  viewportSize;

	
	cl_float3 eyePos, viewDir, up, viewPortStart, viewStep;
	cl_float eyePlaneDist, fov; 
};

cl_renderinfo convert(renderinfo info);

#endif //_OPENCL_RENDER_INFO_H