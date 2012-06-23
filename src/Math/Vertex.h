#ifndef _VERTEX_H
#define _VERTEX_H

#include "Vector.h"

struct vertex {
    public:
                                 vertex(){};
        explicit                 vertex(const float4& position, 
                                        const float4& normal = float4(0.0f), 
                                        const float4& colour = float4(1.0f),
                                        const float2& texCoord = float2(0.0f)) 
                                 :  m_position(position), 
                                    m_normal(normal), 
                                    m_colour(colour),
                                    m_textCoord(texCoord){};
                                    
                                 vertex(const vertex& other) 
                                 :  m_position(other.m_position), 
                                    m_normal(other.m_normal), 
                                    m_colour(other.m_colour),
                                    m_textCoord(other.m_textCoord){};
        
        inline float4            getPosition() const {
            return m_position;
        }
        
        inline float4            getNormal() const {
            return m_normal;
        }
        
        inline float4            getColour() const {
            return m_colour;
        }
        
        inline float2            getTexCoord() const {
            return m_textCoord;
        }
        
        inline void              setPosition(const float4& position) {
            m_position = position;
        }
        
        inline void              setNormal(const float4& normal) {
            m_normal = direction(normal);
        }
        
        inline void              setColour(const float4& colour) {
            m_colour = colour;
        }
        
        inline void              setTexCoord(const float2& texCoord) {
            m_textCoord = texCoord;
        }
        
        inline void              setTexCoord(const float4& texCoord) {
            m_textCoord = float2(texCoord.getX(), texCoord.getY());
        }
        
        inline vertex&           operator=(const vertex& rhs){
            if(this != &rhs) {
                m_position = rhs.m_position;
                m_normal = rhs.m_normal;
                m_colour = rhs.m_colour;
                m_textCoord = rhs.m_textCoord;
            }
            return *this;
        }
        
    private:
        float4                   m_position;
        float4                   m_normal;
        float4                   m_colour;
        float2                   m_textCoord;
};

#endif //_VERTEX_H