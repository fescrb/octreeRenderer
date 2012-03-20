#ifndef _LINE_H
#define _LINE_H

#include "Vector.h"

struct line {
    public:
                             line() {}
        explicit             line(const float4& origin, const float4& direction) : m_origin(origin), m_direction(direction) {}
                             line(const line& other) : m_origin(other.m_origin), m_direction(other.m_direction) {}
        
        
        inline float4        getPositionAt(const float& scalar) const{
            return m_origin + (m_direction * scalar);
        }
        
        inline float4       getOrigin() const {
            return m_origin;
        }
        
        inline float4        getDirection() const {
            return m_direction;
        }
        
        inline void          setOrigin(const float4& origin) {
            m_origin = origin;
        }
        
        inline void          setDirection(const float4& direction) {
            m_direction = direction;
        }
        
    private:
        float4               m_origin;
        float4               m_direction;
};

#endif //_LINE_H