#ifndef _DEVICE_H
#define _DEVICE_H

#include "RenderInfo.h"
#include "Vector.h"


#ifdef _LINUX
	#include <GL/gl.h>
#endif //_LINUX

#ifdef _OSX
	#include <OpenGL/gl.h>
#endif //_OSX

class DeviceInfo;
class OctreeSegment;

class Device {
    public:
		explicit		 Device();
        virtual 		~Device();

        virtual void	 printInfo() = 0;

        /**
         * We clear the framebuffer if we needen't generate it
         * @param The dimensions of the required framebuffer.
         */
        virtual void     makeFrameBuffer(int2 size) = 0;
        virtual void 	 sendData(OctreeSegment* segment) = 0;
        virtual void	 render(float2 start, float2 size, RenderInfo &info) = 0;
        /**
         * Returns the framebuffer as a texture. NOTE: we always
         * assume that the target OpenGL context is CURRENT.
         * @return The OpenGL texture id containing the framebuffer.
         */
        virtual GLuint   getFrameBuffer() = 0;
        virtual char    *getFrame() = 0;
    protected:

        DeviceInfo		*m_pDeviceInfo;
};

#endif // _DEVICE_H
