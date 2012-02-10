#ifndef _RENDER_INFO_H
#define _RENDER_INFO_H

#include "Vector.h"

struct RenderInfo {
	// Scene info. Hard coded. Scene boundaries go from -256 to 256.
	//float leftCorner[3];
	//float size;
	int   maxOctreeDepth;

	// Projection Info.
	int	  resolution[2];

	// Camera info.
	float3 eyePos;
	float3 viewDir;
	float eyePlaneDist, fov; // Distance from near plane + field of view angle.
};

#endif //_RENDER_INFO_H
