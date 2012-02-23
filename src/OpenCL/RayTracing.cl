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

bool no_children(__global char* address) {
    return !address[0];
}

__global char* find_collission(__global char* octree, float3 origin, float3 direction, float t) {

	float half_size = 256.0f;

	float3 corner_far = (float3)(direction.x >= 0 ? half_size : -half_size,
								 direction.y >= 0 ? half_size : -half_size,
								 direction.z >= 0 ? half_size : -half_size);

	float3 corner_close = (float3)(-corner_far.x,-corner_far.y,-corner_far.z);

	float t_min = max_component((corner_close - origin) / direction);
	float t_max = min_component((corner_far - origin) / direction);
	float t_out = t_max;
			
	/* If we are out */
	if(t < t_min)
		t = t_min;
			
	__global char* curr_address = octree;
	float3 voxelCentre = (float3)(0.0f);
	bool collission = false;	
	int curr_index = 0;
	struct stack short_stack[5];
			
	/* We are out of the volume and we will never get to it. */
	if(t > t_max) {
		collission = true;
		curr_address = 0; 
	}

	while(!collission) {
        collission = true;

        if(no_children(curr_address)) {
            collission = true;
        } else {
            /*If we are inside the node*/
            if(t_min <= t && t < t_max) {
                float3 rayPos = origin + (direction * t);
                
                char xyz_flag = makeXYZFlag(rayPos, voxelCentre);
                float nodeHalfSize = fabs((corner_far-voxelCentre)[0])/2.0f;
                
                float3 tmpNodeCentre( xyz_flag & 1 ? voxelCentre[0] + nodeHalfSize : voxelCentre[0] - nodeHalfSize,
                                      xyz_flag & 2 ? voxelCentre[1] + nodeHalfSize : voxelCentre[1] - nodeHalfSize,
                                      xyz_flag & 4 ? voxelCentre[2] + nodeHalfSize : voxelCentre[2] - nodeHalfSize);
                
                float3 tmp_corner_far(d[0] >= 0 ? tmpNodeCentre[0] + nodeHalfSize : tmpNodeCentre[0] - nodeHalfSize,
                                      d[1] >= 0 ? tmpNodeCentre[1] + nodeHalfSize : tmpNodeCentre[1] - nodeHalfSize,
                                      d[2] >= 0 ? tmpNodeCentre[2] + nodeHalfSize : tmpNodeCentre[2] - nodeHalfSize);
                
                float tmp_max = min((tmp_corner_far - rayPos) / d);
                
                if(nodeHasChildAt(rayPos, voxelCentre, curr_address)) {
                    /* If the voxel we are at is not empty, go down. */
                    
                    curr_index = push(stack, curr_index, curr_address, corner_far, voxelCentre, t_min, t_max);
                    
                    curr_address = getChild(rayPos,voxelCentre,curr_address);
                    
                    corner_far = tmp_corner_far;
                    voxelCentre = tmpNodeCentre;
                    corner_close =  float3(d[0] >= 0 ? voxelCentre[0] - nodeHalfSize : voxelCentre[0] + nodeHalfSize,
                                            d[1] >= 0 ? voxelCentre[1] - nodeHalfSize : voxelCentre[1] + nodeHalfSize,
                                            d[2] >= 0 ? voxelCentre[2] - nodeHalfSize : voxelCentre[2] + nodeHalfSize);
                    t_max = tmp_max;
                    t_min = max((corner_close - rayPos) / d);
                    
                } else {
                    /* If the child is empty, we step the ray. */
                    t = tmp_max;
                }
            } else {
                /* We are outside the node. Pop the stack */
                curr_index--;
                if(curr_index>=0) {
                    /* Pop that stack! */
                    corner_far = stack[curr_index].far_corner;
                } else {
                    /* We are outside the volume. */
                    curr_address = 0;
                    collission = true;
                }
            }
        }
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