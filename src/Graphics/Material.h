#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "Vector.h"

#include "Texture.h"

struct material {
                     material() : diffuse(float4(1.0f)), texture(0){}
    float4           diffuse;
    Texture         *texture;
};

#endif //_MATERIAL_H