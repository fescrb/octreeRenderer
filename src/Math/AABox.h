#ifndef _A_A_BOX_H
#define _A_A_BOX_H

#include "Mesh.h"

struct aabox {
    public:
        explicit        aabox();
        mesh            cull(mesh);
        
    private:
        float4          m_corner;
        float3          m_sizes;
};

#endif //_A_A_BOX_H