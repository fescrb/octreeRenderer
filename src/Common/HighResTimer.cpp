#include "HighResTimer.h"

#include <time.h>

#define NANOSECS_PER_SECOND 1000000000.0f

high_res_timer::high_res_timer() {
	reset();
}

void high_res_timer:: reset() {
	timespec time;
	clock_gettime(CLOCK_REALTIME, &time);
	m_seconds = time.tv_sec;
	m_seconds += ((double)time.tv_nsec/(double)NANOSECS_PER_SECOND);
}