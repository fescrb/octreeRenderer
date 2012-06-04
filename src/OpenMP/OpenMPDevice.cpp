#include "OpenMPDevice.h"

#include <cstdio>

void OpenMPDevice::renderTask(int index, renderinfo *info) { 
    //printf("",m_pHeader[1]);
    
    rect window = m_tasks[index];
    int2 start = window.getOrigin();
    int2 size = window.getSize();
    
	int2 end = start+size;
    
    //printf("start %d %d end %d %d\n", start.getX(), start.getY(), end.getX(), end.getY());

    #pragma omp parallel for
    for(int y = start[1]/RAY_BUNDLE_WINDOW_SIZE; y < end[1]/RAY_BUNDLE_WINDOW_SIZE; y++) {
        #pragma omp parallel for
        for(int x = start[0]/RAY_BUNDLE_WINDOW_SIZE; x < end[0]/RAY_BUNDLE_WINDOW_SIZE; x++) {
            traceRayBundle(x, y, 8, info);
            //printf("done %d %d\n", x, y);
        }
    }
    
	#pragma omp parallel for
	for(int y = start[1]; y < end[1]; y++) {
		#pragma omp parallel for
		for(int x = start[0]; x < end[0]; x++) {
			traceRay(x, y, info);
		}
	}
}