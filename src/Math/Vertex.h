#ifndef _VERTEX_H
#define _VERTEX_H

#include "Vector.h"

struct vertex {
    public:
        explicit                 vertex(float4 position, 
                                        float4 normal, 
                                        float4  colour) 
                                 :  m_position(position), 
                                    m_normal(normal), 
                                    m_colour(colour) {};
        
        inline float4            getPosition() {
            return m_position;
        }
        
        inline float4            getNormal() {
            return m_normal;
        }
        
        inline float4            getColour() {
            return m_colour;
        }
        
    private:
        float4                   m_position;
        float4                   m_normal;
        float4                   m_colour;
};

#endif //_VERTEX_H