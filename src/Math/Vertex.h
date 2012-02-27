#ifndef _VERTEX_H
#define _VERTEX_H

#include "Vector.h"

struct vertex {
    public:
                                 vertex(){};
        explicit                 vertex(const float4& position, 
                                        const float4& normal, 
                                        const float4&  colour) 
                                 :  m_position(position), 
                                    m_normal(normal), 
                                    m_colour(colour) {};
                                    
                                 vertex(const vertex& other) 
                                 :  m_position(other.m_position), 
                                    m_normal(other.m_normal), 
                                    m_colour(other.m_colour) {};
        
        inline float4            getPosition() {
            return m_position;
        }
        
        inline float4            getNormal() {
            return m_normal;
        }
        
        inline float4            getColour() {
            return m_colour;
        }
        
        inline vertex&           operator=(const vertex& rhs){
            if(this != &rhs) {
                m_position = rhs.m_position;
                m_normal = rhs.m_normal;
                m_colour = rhs.m_colour;
            }
            return *this;
        }
        
    private:
        float4                   m_position;
        float4                   m_normal;
        float4                   m_colour;
};

#endif //_VERTEX_H