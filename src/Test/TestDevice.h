#ifndef _TEST_DEVICE_H
#define _TEST_DEVICE_H

#include "Device.h"

class TestDevice
:	public Device {
    public:
		explicit		 TestDevice();
        virtual 		~TestDevice();

        void			 printInfo();
        void             makeFrameBuffer(int2 size);
        void 			 sendData(OctreeSegment* segment);
        void			 render(int2 start, int2 size, renderinfo *info);
        GLuint           getFrameBuffer();
        char    		*getFrame();

    private:
        char			*m_pOctreeData;
        char			*m_pFrame;
        int2             m_frameBufferResolution;
        GLuint           m_texture;
        
        void			 setFramePixel(int x, int y,
									   char red, char green, char blue);
};

#endif // _TEST_DEVICE_H
