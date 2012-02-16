#ifndef _VECTOR_2_H
#define _VECTOR_2_H

#include "BasicTypes.h"
#include <cmath>
#include <cstdlib>

template <class t>
struct vector2{
	public:
		explicit 		 vector2(): m_x(0), m_y(0){}
		explicit 		 vector2(t value): m_x(value), m_y(value){}
		explicit 		 vector2(t x, t y): m_x(x), m_y(y){}
						 vector2(const vector2 &vector): m_x(vector.m_x), m_y(vector.m_y){}

		inline vector2 & operator=(const vector2 &rhs){
			if(this != &rhs){
				m_x = rhs.m_x;
				m_y = rhs.m_y;
			}
			return *this;
		}

		inline vector2   operator+(const vector2 &rhs) const{
			return vector2(m_x+rhs.m_x, m_y+rhs.m_y);
		}

		inline vector2   operator-(const vector2 &rhs) const{
			return vector2(m_x-rhs.m_x, m_y-rhs.m_y);
		}

		inline vector2   operator*(const t &rhs) const{
			return vector2(m_x*rhs, m_y*rhs);
		}

		inline vector2   operator*(const vector2 &rhs) const{
			return vector2(m_x*rhs.m_x,m_y*rhs.m_y);
		}

		inline vector2   operator/(const t &rhs) const{
			return vector2(m_x/rhs,m_y/rhs);
		}
		
		inline vector2   operator/(const vector2 &rhs) const{
			return vector2(m_x/rhs.m_x,m_y/rhs.m_y);
		}

		inline bool      operator==(const vector2 &rhs)const{
			return m_x==rhs.m_x && m_y==rhs.m_y;
		}

		inline t	     operator[](const I32 &index)const{
			switch(index){
				case 0:
					return m_x;
				case 1:
					return m_y;
				default:
					return 0;
			}
		}

		inline vector2	 replicateX() const{
			return vector2(m_x);
		}

		inline vector2 	 replicateY() const{
			return vector2(m_y);
		}

		inline void   setX(const t newX) {
			m_x = newX;
		}

		inline void   setY(const t newY) {
			m_y = newY;
		}

		inline t      getX() const {
			return m_x;
		}

		inline t      getY() const {
			return m_y;
		}

		inline void 	 copyToArray(t *array) {
			array[0] = m_x;
			array[1] = m_y;
		}
		
		inline vector2<t> neg() {
			return vector2<t>(-m_x, -m_y);
		}

		template<class t2>
		friend t2          sum(const vector2<t2> &vector);
		template<class t2>
		friend vector2<t2> cross(const vector2<t2> &lhs, const vector2<t2> &rhs);

	private:
		t m_x, m_y;
};

template <class t>
inline vector2<t> add(const vector2<t> &lhs, const vector2<t> &rhs){
	return lhs + rhs;
}

template <class t>
inline vector2<t> sub(const vector2<t> &lhs, const vector2<t> &rhs){
	return lhs - rhs;
}

template <class t>
inline vector2<t> mul(const vector2<t> &lhs, const t &rhs){
	return lhs * rhs;
}

template <class t>
inline vector2<t> mul(const vector2<t> &lhs, const vector2<t> &rhs){
	return lhs * rhs;
}

template <class t>
inline vector2<t> div(const vector2<t> &lhs, const t &rhs){
	return lhs / rhs;
}

template <class t>
inline t sum(const vector2<t> &vector){
	return vector.getX() + vector.getY();
}

template <class t>
inline t mag(const vector2<t> &vector){
	return sqrt(fabs(sum(mul(vector,vector))));
}

#endif //_VECTOR_2_H
