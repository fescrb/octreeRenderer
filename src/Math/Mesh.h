#ifndef _MESH_H
#define _MESH_H

#include "Triangle.h"

#include <vector>

struct mesh {
    public:
        explicit                 mesh();
        
    private:
        std::vector<triangle>    triangles;
};

#endif //_MESH_H