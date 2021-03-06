#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include "Vertex.h"

#include <cstdio>

#include "Texture.h"

struct triangle {
    public:
                         triangle() : m_texture(0) {};
        explicit         triangle(const vertex& vertex1, 
                                  const vertex& vertex2, 
                                  const vertex& vertex3,
                                  Texture *texture = 0) 
                         :  m_vert0(vertex1), 
                            m_vert1(vertex2), 
                            m_vert2(vertex3),
                            m_texture(texture){};
        
                         triangle(const triangle& other) 
                         :  m_vert0(other.m_vert0), 
                            m_vert1(other.m_vert1), 
                            m_vert2(other.m_vert2),
                            m_texture(other.m_texture) {};
        
        
        inline vertex&   operator[](const int& index) {
            switch(index) {
                case 0:
                    return m_vert0;
                case 1:
                    return m_vert1;
                case 2:
                    return m_vert2;
                default:
                    return m_vert0;
            }
        }
        
        inline vertex    getVertex(const int& index) const {
            switch(index) {
                case 0:
                    return m_vert0;
                case 1:
                    return m_vert1;
                case 2:
                    return m_vert2;
                default:
                    return m_vert0;
            }
        }
        
        // Broken
        inline vertex    interpolate(const float4& position) const{
            float3 scales;
            
            scales.setX(mag(position - getVertex(1).getPosition()) / mag(getVertex(1).getPosition()-getVertex(0).getPosition()));
            scales.setY(mag(position - getVertex(2).getPosition()) / mag(getVertex(2).getPosition()-getVertex(1).getPosition()));
            scales.setZ(mag(position - getVertex(0).getPosition()) / mag(getVertex(0).getPosition()-getVertex(2).getPosition()));
            
            scales = normalize(scales);
            
            vertex new_vertex;
            
            new_vertex.setPosition(position);
            new_vertex.setNormal((getVertex(0).getNormal()*scales[0])+(getVertex(1).getNormal()*scales[1])+(getVertex(2).getNormal()*scales[2]));
            new_vertex.setColour((getVertex(0).getColour()*scales[0])+(getVertex(1).getColour()*scales[1])+(getVertex(2).getColour()*scales[2]));
            
            return new_vertex;
        }
        
        inline void      generateNormals() {
            // Calculate face normal
            float4 temp_scnd = m_vert1.getPosition() - m_vert0.getPosition();
            float4 temp_thrd = m_vert2.getPosition() - m_vert0.getPosition();
            
            float4 normal = cross(temp_thrd, temp_scnd);
            double magnitude = mag(normal);
            normal = normalize(normal);
            //printf("mag %f norm %f %f %f %f\n", magnitude, normal[0], normal[1], normal[2], normal[3]);
            
            m_vert0.setNormal(normal);
            m_vert1.setNormal(normal);
            m_vert2.setNormal(normal);
        }
        
        inline float4    getAverageNormal() const{
            return (m_vert0.getNormal() + m_vert1.getNormal() + m_vert2.getNormal())/3.0f;
        }
        
        inline float4    getAverageColour() const{
            if(!m_texture) {
                //printf("text %p\n", m_texture);
                return (m_vert0.getColour() + m_vert1.getColour() + m_vert2.getColour())/3.0f;
            } else {
                float2 avg_tex_coord = (m_vert0.getTexCoord() + m_vert1.getTexCoord() + m_vert2.getTexCoord())/3.0f;
                return m_texture->getColourAt(avg_tex_coord);
            }
        }
        
        double            getSurfaceArea() const {
            float4 l1 = m_vert1.getPosition() - m_vert0.getPosition();
            float4 l2 = m_vert2.getPosition() - m_vert0.getPosition();
            double l_l1 = mag(l1);
            double l_l2 = mag(l2);
            
            double cos_angle = (dot(l1,l2)) / (l_l1*l_l2);
            if(cos_angle == INFINITY || cos_angle == -INFINITY)
                return 0;
            while(cos_angle > 1.0f)
                cos_angle-=1.0f;
            while(cos_angle < -1.0f)
                cos_angle+=1.0f;
            double angle = acos(cos_angle);
            double opposite = sin(angle)*l_l1;
            
            return (opposite*l_l2)/2.0f;
        }
        
        void             setTexture(Texture* texture) {
            m_texture = texture;
        }
        
        Texture         *getTexture() const {
            return m_texture;
        }
        
        void             render() const;
        
    private:
        vertex           m_vert0, m_vert1, m_vert2;
        Texture         *m_texture;
};

#endif //_TRIANGLE_H