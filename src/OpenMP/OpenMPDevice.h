#ifndef _OPENMP_DEVICE_H
#define _OPENMP_DEVICE_H

#include "SerialDevice.h"

class OpenMPDevice 
:   public  SerialDevice {
	public:
		void		 renderTask(int index, renderinfo *info);
};

#endif //_OPENMP_DEVICE_H
