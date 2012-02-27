#ifndef _OPENMP_DEVICE_H
#define _OPENMP_DEVICE_H

#include "SerialDevice.h"

class OpenMPDevice 
:   public  SerialDevice {
	public:
		void		 render(int2 start, int2 size, renderinfo *info);
};

#endif //_OPENMP_DEVICE_H
