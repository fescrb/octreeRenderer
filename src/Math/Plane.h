#ifndef _PLANE_H
#define _PLANE_H

#include "Vector.h"

#include "Triangle.h"

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
        
        inline bool              liesOnPlane(const float4& point) {
            return 0 == isInFront(point);
        }
        
        inline bool              insidePlane(const float4& point) {
            return isInFront(point) >= 0;
        }
        
        /**
            * @return Bigger than 0 if point is in front. 0 iff point
            * lies on the plane. And less than 0 if it's behind.
            */
        inline F32               isInFront(const float4& point) {
            return dot(m_normal, point-m_pointInPlane);
        }
        
        std::vector<triangle>    cull(const triangle& triangleToCull) {
            std::vector<triangle> result(0);
            
            bool within_pane[3] = { 
                insidePlane(triangleToCull.getVertex(0).getPosition()),
                insidePlane(triangleToCull.getVertex(1).getPosition()),
                insidePlane(triangleToCull.getVertex(2).getPosition())
            };
            
            if(within_pane[0] && within_pane[1] && within_pane[2]) {
                result.push_back(triangleToCull);
            } else if (within_pane[0] || within_pane[1] || within_pane[2]) {
                
            }
            
            return result;
        }
    
    private:
        float4                   m_pointInPlane;
        float4                   m_normal;
};

#endif //_PLANE_H