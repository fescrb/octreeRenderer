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
					 cl_renderinfo(){}
	explicit 		 cl_renderinfo(cl_int maxDept, 
								   cl_int2 viewport,
								   cl_float3 eye,
								   cl_float3 dir,
								   cl_float3 upV,
								   cl_float3 viewportS,
								   cl_float3 viewS,
								   cl_float eyePlaneD,
								   cl_float fovR )
	: 	maxOctreeDepth(maxDept), 
		viewportSize(viewport), 
		eyePos(eye), 
		viewDir(dir), 
		up(upV), 
		viewPortStart(viewportS), 
		viewStep(viewS), 
		eyePlaneDist(eyePlaneD), 
		fov(fovR){}
	
	cl_int   maxOctreeDepth;
    
    cl_int2  viewportSize;

	
	cl_float3 eyePos, viewDir, up, viewPortStart, viewStep;
	cl_float eyePlaneDist, fov; 
};

cl_renderinfo convert(renderinfo info);

#endif //_OPENCL_RENDER_INFO_H