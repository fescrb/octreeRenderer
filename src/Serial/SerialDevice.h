#ifndef _SERIAL_DEVICE_H
#define _SERIAL_DEVICE_H

#include "Device.h"

class SerialDevice
:	public Device {
    public:
		explicit		     SerialDevice();
        virtual 		    ~SerialDevice();

        void			     printInfo();
        void 			     sendData(Bin bin);
        void                 sendHeader(Bin bin);
        void                 makeFrameBuffer(int2 size);
        void                 traceRayBundle(int x, int y, int width, renderinfo* info);
        void                 traceRay(int x, int y, renderinfo* info);
        virtual void	     renderTask(int index, renderinfo *info);
        framebuffer_window   getFrameBuffer();
        unsigned char       *getFrame();

    private:
        char			    *m_pOctreeData;
        char                *m_pHeader;
        GLuint               m_texture;

        void			     setFramePixel(int x, int y,
                                           unsigned char red, unsigned char green, unsigned char blue);
        
        void                 setInfoPixels(int x, int y,
                                           float depth, unsigned char iterations, unsigned char depth_in_octree);
        float                getDepthBufferValue(int x, int y);
        void                 setDepthBufferValue(int x, int y, float value);
};

#endif // _SERIAL_DEVICE_H
