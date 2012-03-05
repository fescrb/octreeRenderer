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
        
    private:
        vertex           m_vert0, m_vert1, m_vert2;
};

#endif //_TRIANGLE_H