struct renderinfo{
	int   maxOctreeDepth;
    
    int2  viewportSize;

	float3 eyePos, viewDir, up, viewPortStart, viewStep;
	float eyePlaneDist, fov;
};

struct stack{
	char* address;
	float3 far_corner, node_centre;
	float t_min, t_max;
};

float min_component(float3 vector) {
	float minimum = vector.x < vector.y? vector.x : vector.y;
	return minimum < vector.z ? minimum : vector.z;
}

float max_component(float3 vector) {
	float maximum = vector.x > vector.y? vector.x : vector.y;
	return maximum > vector.z ? maximum : vector.z;
}

char* find_collission(__global char* octree, float3 origin, float3 direction, float t) {

	float half_size = 256.0f;

	float3 corner_far = (float3)(direction.x >= 0 ? half_size : -half_size,
								 direction.y >= 0 ? half_size : -half_size,
								 direction.z >= 0 ? half_size : -half_size);

	float3 corner_close = (float3)(-corner_far.x,-corner_far.y,-corner_far.z);

	float t_min = max_component((corner_close - origin) / direction);
	float t_max = min_component((corner_far - origin) / direction);
	float t_out = t_max;
			
	// If we are out 
	if(t < t_min)
		t = t_min;
			
	char* curr_address = octree;
	float3 voxelCentre = (float3)(0.0f);
	bool collission = false;	
	int curr_index = 0;
	struct stack short_stack[5];
			
	// We are out of the volume and we will never get to it.
	if(t > t_max) {
		collission = true;
		curr_address = 0; // Set to null.
	}

	while(!collission) {
	}

	return curr_address;
}

__kernel void ray_trace(__global char* octree,
                        struct renderinfo info,
                        int widthOfFramebuffer,
                        __global unsigned char* frameBuff) {
    int x = get_global_id(0);
	int y = get_global_id(1);

	float3 o = info.viewPortStart + (info.viewStep * x) + (info.up * y);
    float3 d = o-info.eyePos; 



	int pixel_index = (x*3)+(y*widthOfFramebuffer*3);
	frameBuff[pixel_index + 0] = 255;
	frameBuff[pixel_index + 1] = 255;
	frameBuff[pixel_index + 2] = 255;
}