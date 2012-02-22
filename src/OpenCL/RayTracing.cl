struct renderinfo {
	int   maxOctreeDepth;
    
    int2  viewportSize;

	float3 eyePos, viewDir, up, viewPortStart, viewStep;
	float eyePlaneDist, fov;
};

__kernel void ray_trace(__global char* octree,
                        struct renderinfo info,
                        int widthOfFramebuffer,
                        __global unsigned char* frameBuff) {
    int x = get_global_id(0);
	int y = get_global_id(1);

	int pixel_index = x+(y*widthOfFramebuffer*3);
	frameBuff[pixel_index + 0] = 255;
	frameBuff[pixel_index + 1] = 255;
	frameBuff[pixel_index + 2] = 255;
}