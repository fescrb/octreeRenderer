#ifndef _DEVICE_H
#define _DEVICE_H

#include "Vector.h"

#include "HighResTimer.h"

#ifdef _LINUX
	#include <GL/gl.h>
#endif //_LINUX

#ifdef _OSX
	#include <OpenGL/gl.h>
#endif //_OSX

class DeviceInfo;
class OctreeSegment;
class renderinfo;

class Device {
    public:
		explicit                 Device();
        virtual                 ~Device();
		
		virtual char            *getName();

        virtual void             printInfo() = 0;

        /**
         * We clear the framebuffer if we needen't generate it
         * @param The dimensions of the required framebuffer.
         */
        virtual void             makeFrameBuffer(int2 size) = 0;
        virtual void             sendData(OctreeSegment* segment) = 0;
        virtual void             render(int2 start, int2 size, renderinfo *info) = 0;
        /**
         * Returns the framebuffer as a texture. NOTE: we always
         * assume that the target OpenGL context is CURRENT.
         * @return The OpenGL texture id containing the framebuffer.
         */
        virtual GLuint           getFrameBuffer() = 0;
        virtual char            *getFrame() = 0;
    
        virtual high_res_timer   getRenderTime() = 0;
        virtual high_res_timer   getBufferToTextureTime() = 0;
        
    protected:

        DeviceInfo               *m_pDeviceInfo;
};

#endif // _DEVICE_H
