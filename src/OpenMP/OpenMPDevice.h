#ifndef _OPENMP_DEVICE_H
#define _OPENMP_DEVICE_H

#include "SerialDevice.h"

class OpenMPDevice 
:   public  SerialDevice {
	public:
		void		 render(rect *window, renderinfo *info);
};

#endif //_OPENMP_DEVICE_H
