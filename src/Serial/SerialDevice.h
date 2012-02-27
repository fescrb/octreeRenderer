#ifndef _SERIAL_DEVICE_H
#define _SERIAL_DEVICE_H

#include "Device.h"

class SerialDevice
:	public Device {
    public:
		explicit		 SerialDevice();
        virtual 		~SerialDevice();

        void			 printInfo();
        void             makeFrameBuffer(int2 size);
        void 			 sendData(OctreeSegment* segment);
        void             traceRay(int x, int y, renderinfo* info);
        virtual void	 render(int2 start, int2 size, renderinfo *info);
        GLuint           getFrameBuffer();
        char    		*getFrame();
    
        high_res_timer   getRenderTime();
        high_res_timer   getBufferToTextureTime();

	protected:
		high_res_timer   m_renderStart;
        high_res_timer   m_renderEnd;
        high_res_timer   m_transferStart;
        high_res_timer   m_transferEnd;
		
    private:
        char			*m_pOctreeData;
        char			*m_pFrame;
        int2             m_frameBufferResolution;
        GLuint           m_texture;
        
        void			 setFramePixel(int x, int y,
									   char red, char green, char blue);
};

#endif // _SERIAL_DEVICE_H
