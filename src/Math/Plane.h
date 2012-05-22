#ifndef _PLANE_H
#define _PLANE_H

#include "Vector.h"

#include "Triangle.h"
#include "Line.h"

#include <vector>

struct plane {
    public:
                                 plane() {}
                            
        explicit                 plane(const float4& pointInPlane,
                               const float4& planeNormal )
        :	m_pointInPlane(pointInPlane),
            m_normal(planeNormal) {
        }
        
                                 plane(const plane& other)
        : 	m_pointInPlane(other.m_pointInPlane),
            m_normal(other.m_normal) {
        }
        
        inline bool              liesOnPlane(const float4& point) const {
            return 0 == isInFront(point);
        }
        
        inline bool              insidePlane(const float4& point) const {
            return isInFront(point) >= 0;
        }
        
        /**
            * @return Bigger than 0 if point is in front. 0 iff point
            * lies on the plane. And less than 0 if it's behind.
            */
        inline F32               isInFront(const float4& point) const {
            return dot(m_normal, point-m_pointInPlane);
        }
        
        float4                   getIntersectionPoint(const line& intersecting_line);
        
        std::vector<triangle>    cull(const triangle& triangleToCull);
        
        inline float4            getPointInPlane() const {
            return m_pointInPlane;
        }
        
        inline float4            getNormal() const {
            return m_normal;
        }
        
    private:
        float4                   m_pointInPlane;
        float4                   m_normal;
};

#endif //_PLANE_H