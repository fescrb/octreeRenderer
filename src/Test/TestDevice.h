#ifndef _TEST_DEVICE_H
#define _TEST_DEVICE_H

#include "Device.h"

class TestDevice
:	public Device {
    public:
		explicit		 TestDevice();
        virtual 		~TestDevice();

        void			 printInfo();
        void 			 sendData(char* data);
        void			 render(RenderInfo &info);
        char    		*getFrame();

    private:
        char			*m_pOctreeData;
        char			*m_pFrame;
};

#endif // _TEST_DEVICE_H
