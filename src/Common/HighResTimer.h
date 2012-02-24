#ifndef _HIGH_RES_TIMER_H
#define _HIGH_RES_TIMER_H

struct high_res_timer {
	public:
						 high_res_timer();
		void 			 reset();
		
		inline high_res_timer operator-(const high_res_timer rhs) {
			high_res_timer timer;
			timer.m_seconds = m_seconds -rhs.m_seconds;
			return timer;
		}
		
		inline operator  double() {
			return m_seconds;
		}
		
	private:
		
		double 			 m_seconds;
};

#endif //_HIGH_RES_TIMER_H