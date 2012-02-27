#ifndef _TRIANGLE_H
#define _TRIANGLE_H

#include "Vector.h"

struct triangle {
    public:
        explicit         triangle();
        
    private:
        float4           m_vert1, m_vert2, m_vert3;
};

#endif //_TRIANGLE_H