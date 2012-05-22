#ifndef _LINE_H
#define _LINE_H

#include "Vector.h"
#include "Vertex.h"

struct line {
    public:
                             line() {}
        explicit             line(const float4& origin, const float4& direction) : m_origin(origin), m_direction(direction) {}
        explicit             line(const vertex& v1, const vertex& v2) : m_origin(v1.getPosition()), m_direction(v2.getPosition()-v1.getPosition()) {}
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
        
        /**
         * Prerequisite: position is in the line
         */
        inline float         getScalarAt(const float4& position) const {
            return ((position-m_origin)/m_direction)[0];
        }
        
        inline void          setOrigin(const float4& origin) {
            m_origin = origin;
        }
        
        inline void          setDirection(const float4& direction) {
            m_direction = direction;
        }
        
        static vertex        linearInterpolation(const vertex& v1,const vertex& v2, const float4& position) {
            float distance_1 = mag(v1.getPosition()-position);
            float distance_2 = mag(v2.getPosition()-position);
            float total = distance_1+distance_2;
            distance_1/=total;
            distance_2/=total;
            return vertex(position, v1.getNormal()*distance_2 + v2.getNormal()*distance_1, v1.getColour()*distance_2 + v2.getColour()*distance_1);
        }
        
    private:
        float4               m_origin;
        float4               m_direction;
};

#endif //_LINE_H