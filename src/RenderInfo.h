#ifndef _RENDER_INFO_H
#define _RENDER_INFO_H

struct RenderInfo {
	// Scene info. Makes no difference on rendering, will probably be hard-coded.
	float leftCorner[3];
	float size;

	// Projection Info.
	int	  resolution[2];

	// Camera info.
	float eyePos[3];
	float viewDir[3];
	float eyePlaneDist, fov;
};

#endif //_RENDER_INFO_H
