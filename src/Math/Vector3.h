#ifndef _VECTOR_3_H
#define _VECTOR_3_H

#include "BasicTypes.h"
#include <cmath>
#include <cstdlib>

template <class t>
struct vector3{
	public:
		explicit 		 vector3(): m_x(0), m_y(0), m_z(0){}
		explicit 		 vector3(t value): m_x(value), m_y(value), m_z(value){}
		explicit 		 vector3(t x, t y, t z): m_x(x), m_y(y), m_z(z){}
						 vector3(const vector3 &vector): m_x(vector.m_x), m_y(vector.m_y), m_z(vector.m_z){}

		inline vector3 & operator=(const vector3 &rhs){
			if(this != &rhs){
				m_x = rhs.m_x;
				m_y = rhs.m_y;
				m_z = rhs.m_z;
			}
			return *this;
		}

		inline vector3   operator+(const vector3 &rhs) const{
			return vector3(m_x+rhs.m_x, m_y+rhs.m_y, m_z+rhs.m_z);
		}

		inline vector3   operator-(const vector3 &rhs) const{
			return vector3(m_x-rhs.m_x, m_y-rhs.m_y, m_z-rhs.m_z);
		}

		inline vector3   operator*(const t &rhs) const{
			return vector3(m_x*rhs, m_y*rhs, m_z*rhs);
		}

		inline vector3   operator*(const vector3 &rhs) const{
			return vector3(m_x*rhs.m_x,m_y*rhs.m_y,m_z*rhs.m_z);
		}

		inline vector3   operator/(const t &rhs) const{
			return vector3(m_x/rhs,m_y/rhs,m_z/rhs);
		}
		
		inline vector3   operator/(const vector3 &rhs) const{
			return vector3(m_x/rhs.m_x,m_y/rhs.m_y,m_z/rhs.m_z);
		}

		inline bool      operator==(const vector3 &rhs)const{
			return m_x==rhs.m_x && m_y==rhs.m_y && m_z==rhs.m_z;
		}

		inline t	     operator[](const I32 &index)const{
			switch(index){
				case 0:
					return m_x;
				case 1:
					return m_y;
				case 2:
					return m_z;
				default:
					return 0;
			}
		}

		inline vector3	 replicateX() const{
			return vector3(m_x);
		}

		inline vector3 	 replicateY() const{
			return vector3(m_y);
		}

		inline vector3 	 replicateZ() const{
			return vector3(m_z);
		}

		inline void   setX(const t newX) {
			m_x = newX;
		}

		inline void   setY(const t newY) {
			m_y = newY;
		}

		inline void   setZ(const t newZ) {
			m_z = newZ;
		}

		inline t      getX() const {
			return m_x;
		}

		inline t      getY() const {
			return m_y;
		}

		inline t      getZ() const {
			return m_z;
		}

		inline void 	 copyToArray(t *array) {
			array[0] = m_x;
			array[1] = m_z;
			array[2] = m_y;
		}
		
		inline vector3<t> neg() {
			return vector3<t>(-m_x, -m_y, -m_z);
		}

		template<class t2>
		friend t2          sum(const vector3<t2> &vector);
		template<class t2>
		friend vector3<t2> cross(const vector3<t2> &lhs, const vector3<t2> &rhs);

	private:
		t m_x, m_y, m_z;
};

template <class t>
inline vector3<t> add(const vector3<t> &lhs, const vector3<t> &rhs){
	return lhs + rhs;
}

template <class t>
inline vector3<t> sub(const vector3<t> &lhs, const vector3<t> &rhs){
	return lhs - rhs;
}

template <class t>
inline vector3<t> mul(const vector3<t> &lhs, const t &rhs){
	return lhs * rhs;
}

template <class t>
inline vector3<t> mul(const vector3<t> &lhs, const vector3<t> &rhs){
	return lhs * rhs;
}

template <class t>
inline vector3<t> div(const vector3<t> &lhs, const t &rhs){
	return lhs / rhs;
}

template <class t>
inline t sum(const vector3<t> &vector){
	return vector.getX() + vector.getY() + vector.getZ();
}

template <class t>
inline t dot(const vector3<t> &lhs, const vector3<t> &rhs){
	return sum(mul(lhs,rhs));
}

template <class t>
inline t mag(const vector3<t> &vector){
	return sqrt(fabs(dot(vector,vector)));
}

template <class t>
vector3<t> cross(const vector3<t> &lhs, const vector3<t> &rhs) {
return vector3<t>((lhs.getY()*rhs.getZ()) - (lhs.getZ()*rhs.getY()),
				-((lhs.getX()*rhs.getZ()) - (lhs.getZ()*rhs.getX())),
				  (lhs.getX()*rhs.getY()) - (lhs.getY()*rhs.getX()));
}

#endif //_VECTOR_3_H
