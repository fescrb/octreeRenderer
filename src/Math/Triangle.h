#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include "Vertex.h"

struct triangle {
    public:
                         triangle(){};
        explicit         triangle(const vertex& vertex1, 
                                  const vertex& vertex2, 
                                  const vertex& vertex3) 
                         :  m_vert0(vertex1), 
                            m_vert1(vertex2), 
                            m_vert2(vertex3) {};
        
                         triangle(const triangle& other) 
                         :  m_vert0(other.m_vert0), 
                            m_vert1(other.m_vert1), 
                            m_vert2(other.m_vert2) {};
        
        
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
            
            m_vert0.setNormal(normal);
            m_vert1.setNormal(normal);
            m_vert2.setNormal(normal);
        }
        
    private:
        vertex           m_vert0, m_vert1, m_vert2;
};

#endif //_TRIANGLE_H