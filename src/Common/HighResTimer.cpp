#include "HighResTimer.h"

#ifdef _LINUX
    #include <time.h>
#endif //_LINUX

#ifdef _OSX
    #include <mach/mach_time.h>
#endif //_OSX

#define NANOSECS_PER_SECOND 1000000000.0f

high_res_timer::high_res_timer() {
	reset();
}

void high_res_timer:: reset() {
#ifdef _LINUX
	timespec time;
	clock_gettime(CLOCK_REALTIME, &time);
	m_seconds = time.tv_sec;
	m_seconds += ((double)time.tv_nsec/(double)NANOSECS_PER_SECOND);
#endif //_LINUX
#ifdef _OSX
    uint64_t time = mach_absolute_time();
    mach_timebase_info_data_t time_info;
    kern_return_t error = mach_timebase_info(&time_info);
    m_seconds = (double)time * ((double)NANOSECS_PER_SECOND * (double)time_info.numer / (double) time_info.denom );
#endif //_OSX
}