#ifndef _MESH_H
#define _MESH_H

#include "Triangle.h"

#include <vector>

struct mesh {
    public:
        explicit                 mesh(){};
        
        inline void              appendTriangle(const triangle& triangl) {
            triangles.push_back(triangl);
        }
        
        inline triangle          getTriangle(const int& index) {
            return operator[](index);
        }
        
        inline void              setTriangle(const int& index, const triangle& triangl) {
            operator[](index) = triangl;
        }
        
        inline void              removeTriangle(const int& index) {
            triangles.erase(triangles.begin()+index);
        }
        
        inline triangle&         operator[](const int& index) {
            return triangles[index];
        }
        
    private:
        std::vector<triangle>    triangles;
};

#endif //_MESH_H