#ifndef _VERTEX_H
#define _VERTEX_H

#include "Vector.h"

struct vertex {
    public:
                                 vertex(){};
        explicit                 vertex(const float4& position, 
                                        const float4& normal, 
                                        const float4& colour) 
                                 :  m_position(position), 
                                    m_normal(normal), 
                                    m_colour(colour) {};
                                 
        explicit                 vertex(const float4& position, 
                                        const float4& normal) 
                                 :  m_position(position), 
                                    m_normal(normal), 
                                    m_colour(1.0f) {};
                                    
        explicit                 vertex(const float4& position) 
                                 :  m_position(position), 
                                    m_normal(0.0f), 
                                    m_colour(1.0f) {};
                                    
                                 vertex(const vertex& other) 
                                 :  m_position(other.m_position), 
                                    m_normal(other.m_normal), 
                                    m_colour(other.m_colour) {};
        
        inline float4            getPosition() const {
            return m_position;
        }
        
        inline float4            getNormal() const {
            return m_normal;
        }
        
        inline float4            getColour() const {
            return m_colour;
        }
        
        inline void              setPosition(const float4& position) {
            m_position = position;
        }
        
        inline void              setNormal(const float4& normal) {
            m_normal = normal;
        }
        
        inline void              setColour(const float4& colour) {
            m_colour = colour;
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