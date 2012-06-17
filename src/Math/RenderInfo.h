#ifndef _RENDER_INFO_H
#define _RENDER_INFO_H

#ifndef _OCL
#include "Vector.h"
#endif //_OCL

#include <cstdio>

struct renderinfo {
	// Scene info. Hard coded. Scene boundaries go from -256 to 256.
	//float leftCorner[3];
	//float size;

	// Camera info.
	float3 eyePos, viewDir, up, viewPortStart, viewStep;
	float eyePlaneDist, fov, pixel_half_size; // Distance from near plane + field of view angle.
	
	float3 lightPos;
    float lightBrightness;
    
    void print() {
        printf("---------\n");
        printf("render_info\neyePos %f %f %f\n", eyePos[0], eyePos[1], eyePos[2]);
        printf("viewDir %f %f %f\n", viewDir[0], viewDir[1], viewDir[2]);
        printf("up %f %f %f\n", up[0], up[1], up[2]);
        printf("viewPortStart %f %f %f\n", viewPortStart[0], viewPortStart[1], viewPortStart[2]);
        printf("viewStep %f %f %f\n", viewStep[0], viewStep[1], viewStep[2]);
        printf("eyePlaneDist %f\n", eyePlaneDist);
        printf("fov %f\n", fov);
        printf("lightPos %f %f %f\n", lightPos[0], lightPos[1], lightPos[2]);
        printf("lightBrightness %f\n", lightBrightness);
        printf("---------\n");
    }
};

#endif //_RENDER_INFO_H
