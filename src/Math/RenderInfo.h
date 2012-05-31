#ifndef _RENDER_INFO_H
#define _RENDER_INFO_H

#ifndef _OCL
#include "Vector.h"
#endif //_OCL

struct renderinfo {
	// Scene info. Hard coded. Scene boundaries go from -256 to 256.
	//float leftCorner[3];
	//float size;

	// Camera info.
	float3 eyePos, viewDir, up, viewPortStart, viewStep;
	float eyePlaneDist, fov, pixel_half_size; // Distance from near plane + field of view angle.
	
	float3 lightPos;
    float lightBrightness;
};

#endif //_RENDER_INFO_H
