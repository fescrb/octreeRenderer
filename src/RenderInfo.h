#ifndef _RENDER_INFO_H
#define _RENDER_INFO_H

struct RenderInfo {
	// Scene info. Hard coded. Scene boundaries go from -256 to 256.
	//float leftCorner[3];
	//float size;

	// Projection Info.
	int	  resolution[2];

	// Camera info.
	float eyePos[3];
	float viewDir[3];
	float eyePlaneDist, fov; // Distance from near plane + field of view angle.
};

#endif //_RENDER_INFO_H
