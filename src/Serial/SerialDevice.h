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
        void                 traceBeam(int x, int y, int width, renderinfo* info);
        void                 traceRay(int x, int y, renderinfo* info);
        virtual void	     renderTask(int index, renderinfo *info);
        framebuffer_window   getFrameBuffer();
        char    		    *getFrame();

        high_res_timer       getRenderTime();
        high_res_timer       getBufferToTextureTime();

	protected:
		high_res_timer       m_renderStart;
        high_res_timer       m_renderEnd;
        high_res_timer       m_transferStart;
        high_res_timer       m_transferEnd;

    private:
        char			    *m_pOctreeData;
        char                *m_pHeader;
        GLuint               m_texture;

        void			     setFramePixel(int x, int y,
                                           unsigned char red, unsigned char green, unsigned char blue);
        
        void                 setInfoPixels(int x, int y,
                                           float depth, unsigned char iterations, unsigned char depth_in_octree);
};

#endif // _SERIAL_DEVICE_H
