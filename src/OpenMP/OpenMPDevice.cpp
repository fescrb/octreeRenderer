#include "OpenMPDevice.h"

void OpenMPDevice::render(rect *window, renderinfo *info) {
	m_renderStart.reset();
    
    int2 start = window->getOrigin();
    int2 size = window->getSize();
    
	int2 end = start+size;

	#pragma omp parallel for
	for(int y = start[1]; y < end[1]; y++) {
		#pragma omp parallel for
		for(int x = start[0]; x < end[0]; x++) {
			traceRay(start[0]+x, start[1]+y, info);
		}
	}
    
    m_renderEnd.reset();
}