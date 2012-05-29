#ifndef _VECTOR_4_H 
#define _VECTOR_4_H

#include "Vector3.h"

template <class t>
struct vector4{
public:
    explicit 		 vector4(): m_x(0), m_y(0), m_z(0), m_w(0){}
    explicit 		 vector4(t value): m_x(value), m_y(value), m_z(value), m_w(value){}
    explicit 		 vector4(t x, t y, t z, t w): m_x(x), m_y(y), m_z(z), m_w(w){}
    explicit         vector4(vector3<t> v, t w): m_x(v.getX()), m_y(v.getY()), m_z(v.getZ()), m_w(w){}
    template<class t2>
    explicit         vector4(vector4<t2> v): m_x(v.getX()), m_y(v.getY()), m_z(v.getZ()), m_w(v.getW()) {}
                     vector4(const vector4 &vector): m_x(vector.m_x), m_y(vector.m_y), m_z(vector.m_z), m_w(vector.m_w){}
    
    inline vector4 & operator=(const vector4 &rhs){
        if(this != &rhs){
            m_x = rhs.m_x;
            m_y = rhs.m_y;
            m_z = rhs.m_z;
            m_w = rhs.m_w;
        }
        return *this;
    }
    
    inline vector4   operator+(const vector4 &rhs) const{
        return vector4(m_x+rhs.m_x, m_y+rhs.m_y, m_z+rhs.m_z, m_w+rhs.m_w);
    }
    
    inline vector4&  operator+=(const vector4 &rhs) {
        this->operator=(this->operator+(rhs));
        return *this;
    }
    
    inline vector4   operator-(const vector4 &rhs) const{
        return vector4(m_x-rhs.m_x, m_y-rhs.m_y, m_z-rhs.m_z, m_w-rhs.m_w);
    }
    
    inline vector4   operator*(const t &rhs) const{
        return vector4(m_x*rhs, m_y*rhs, m_z*rhs, m_w*rhs);
    }
    
    inline vector4   operator*(const vector4 &rhs) const{
        return vector4(m_x*rhs.m_x,m_y*rhs.m_y,m_z*rhs.m_z,m_w*rhs.m_w);
    }
    
    inline vector4   operator/(const t &rhs) const{
        return vector4(m_x/rhs,m_y/rhs,m_z/rhs, m_w/rhs);
    }
    
    inline vector4   operator/=(const t &rhs) {
        this->operator=(this->operator/(rhs));
        return *this;
    }
    
    inline vector4   operator/(const vector4 &rhs) const{
        return vector4(m_x/rhs.m_x,m_y/rhs.m_y,m_z/rhs.m_z, m_w/rhs.m_w);
    }
    
    inline vector4&  operator/=(const vector4 &rhs) {
        this->operator=(this->operator/(rhs));
        return *this;
    }
    
    inline bool      operator==(const vector4 &rhs)const{
        return m_x==rhs.m_x && m_y==rhs.m_y && m_z==rhs.m_z && m_w==rhs.m_w;
    }
    
    inline bool      operator!=(const vector4 &rhs)const{
        return !(operator==(rhs));
    }
    
    inline t	     operator[](const I32 &index)const{
        switch(index){
            case 0:
                return m_x;
            case 1:
                return m_y;
            case 2:
                return m_z;
            case 3:
                return m_w;
            default:
                return 0;
        }
    }
    
    inline vector4	 replicateX() const{
        return vector4(m_x);
    }
    
    inline vector4 	 replicateY() const{
        return vector4(m_y);
    }
    
    inline vector4 	 replicateZ() const{
        return vector4(m_z);
    }
    
    inline vector4 	 replicateW() const{
        return vector4(m_w);
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
    
    inline void   setW(const t newW) {
        m_w = newW;
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
    
    inline t      getW() const {
        return m_w;
    }
    
    inline void 	 copyToArray(t *array) {
        array[0] = m_x;
        array[1] = m_z;
        array[2] = m_y;
        array[3] = m_w;
    }
    
    inline vector4<t> neg() {
        return vector4<t>(-m_x, -m_y, -m_z, -m_w);
    }
    
    inline operator vector3<t>(){
		vector3<t> ret(m_x, m_y, m_z);
		if(m_w!=0.0f)
			ret = ret/m_w;
		return ret;
	}
    
    template<class t2>
    friend t2          sum(const vector4<t2> &vector);
    template<class t2>
    friend vector4<t2> cross(const vector4<t2> &lhs, const vector4<t2> &rhs);
    
private:
    t m_x, m_y, m_z, m_w;
};

template <class t>
inline vector4<t> add(const vector4<t> &lhs, const vector4<t> &rhs){
	return lhs + rhs;
}

template <class t>
inline vector4<t> sub(const vector4<t> &lhs, const vector4<t> &rhs){
	return lhs - rhs;
}

template <class t>
inline vector4<t> mul(const vector4<t> &lhs, const t &rhs){
	return lhs * rhs;
}

template <class t>
inline vector4<t> mul(const vector4<t> &lhs, const vector4<t> &rhs){
	return lhs * rhs;
}

template <class t>
inline vector4<t> div(const vector4<t> &lhs, const t &rhs){
	return lhs / rhs;
}

template <class t>
inline t sum(const vector4<t> &vector){
	return vector.getX() + vector.getY() + vector.getZ() + vector.getW();
}

template <class t>
inline F32 dot(const vector4<t> &lhs, const vector4<t> &rhs){
	return sum(mul(lhs,rhs));
}

template <class t>
inline t mag(const vector4<t> &vector){
	return sqrt(fabs(dot(vector,vector)));
}


template <class t>
inline vector4<t> normalize(const vector4<t> &vector){
	return vector/mag(vector);
}

template <class t>
vector4<t> cross(const vector4<t> &lhs, const vector4<t> &rhs) {
    return vector4<t>((lhs.getY()*rhs.getZ()) - (lhs.getZ()*rhs.getY()),
                    -((lhs.getX()*rhs.getZ()) - (lhs.getZ()*rhs.getX())),
                      (lhs.getX()*rhs.getY()) - (lhs.getY()*rhs.getX()),
                      lhs.getW()*rhs.getW()
                     );
}

template <class t>
inline vector4<t> direction(const vector3<t> &vector){
	return vector4<t>(vector[0],vector[1],vector[2],0.0f);
}

template <class t>
inline vector4<t> direction(const vector4<t> &vector){
    return vector4<t>(vector[0],vector[1],vector[2],0.0f);
}

template <class t>
inline vector4<t> position(const vector3<t> &vector){
	return vector4<t>(vector[0],vector[1],vector[2],1.0f);
}


#endif //_VECTOR_4_H
