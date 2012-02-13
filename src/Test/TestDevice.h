#ifndef _TEST_DEVICE_H
#define _TEST_DEVICE_H

#include "Device.h"

class TestDevice
:	public Device {
    public:
		explicit		 TestDevice();
        virtual 		~TestDevice();

        void			 printInfo();
        void 			 sendData(char* data, size_t size);
        void			 render(RenderInfo &info);
        char    		*getFrame();

    private:
        char			*m_pOctreeData;
        char			*m_pFrame;
        int				 m_frameBufferResolution[2];
        
        void			 setFramePixel(int x, int y,
									   char red, char green, char blue);
};

#endif // _TEST_DEVICE_H
