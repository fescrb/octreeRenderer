#ifndef _CL_INCLUDES_H
#define _CL_INCLUDES_H

#ifdef _LINUX
    #include <CL/cl.h>
    #include <CL/cl_gl.h>
#endif //_LINUX

#ifdef _OSX
    #include <OpenCL/cl.h>
#endif //_OSX

#endif //_CL_INCLUDES_H