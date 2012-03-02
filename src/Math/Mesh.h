#ifndef _MESH_H
#define _MESH_H

#include "Triangle.h"

#include <vector>

struct mesh {
    public:
                                         mesh(){};
                                         mesh(const mesh& other)
                                         :  m_triangles(other.m_triangles){};
        
        inline void                      appendTriangle(const triangle& triangl) {
            m_triangles.push_back(triangl);
        }
        
        inline void                      appendTriangles(const std::vector<triangle>& triangles) {
            m_triangles.insert(m_triangles.end(), triangles.begin(), triangles.end());
        }
        
        inline triangle                  getTriangle(const int& index) {
            return operator[](index);
        }
        
        inline std::vector<triangle>&    getTriangleList() {
            return m_triangles;
        }
        
        inline void                      setTriangle(const int& index, const triangle& triangl) {
            operator[](index) = triangl;
        }
        
        inline void                      removeTriangle(const int& index) {
            m_triangles.erase(m_triangles.begin()+index);
        }
        
        inline triangle&                 operator[](const int& index) {
            return m_triangles[index];
        }
        
        //inline mesh&                 operator*(const );
        
        /**
         * Returns the 6 outermost vertices that bound the volume in the 
         * x positive and negative directions as well as the y and z.
         * @return The outermost vertices in the following order: {x+, x-, y+, y-, z+. z-}
         */
        std::vector<vertex>              getOuterMostVertices() const;
        
        void                             render() const;
        
    private:
        std::vector<triangle>            m_triangles;
};

#endif //_MESH_H