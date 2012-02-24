#define STACK_SIZE 5

struct renderinfo{
	int   maxOctreeDepth;
    
    int2  viewportSize;

	float3 eyePos, viewDir, up, viewPortStart, viewStep;
	float eyePlaneDist, fov;
};

struct stack{
	global char* address;
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

char makeXYZFlag(float3 rayPos, float3 nodeCentre) {
	float3 flagVector = rayPos - nodeCentre;
	char flag = 0;

	if(flagVector.x >= 0.0f)
		flag |= 1;
	if(flagVector.y >= 0.0f)
		flag |= 2;
	if(flagVector.z >= 0.0f)
		flag |= 4;
	
	return flag;
}

char makeChildFlag(float3 rayPos, float3 nodeCentre) {
	return 0 | (1 << makeXYZFlag(rayPos, nodeCentre));
}

bool nodeHasChildAt(float3 rayPos, float3 nodeCentre, global char* node) {
	return node[0] & makeChildFlag(rayPos,nodeCentre);  
}

global char* getChild(float3 rayPos, float3 nodeCentre, global char* node) {
	int counter = 0;
	bool found = false;
	char xyz_flag = makeXYZFlag(rayPos, nodeCentre);
	char flag = 0 | (1 << xyz_flag);
	for(int i = 0; i <= xyz_flag; i++) 
		if(node[0] & (1 << i))
			counter++;
	node+=(counter*4);
	global int* add_int = (global int*)node;
	return node + (add_int[0]*4);
}

global char* get_attributes(global char* node) {
	global short* addr_short = (global short*)node;
	
	return node + (addr_short[1] * 4);
}

int push(struct stack* short_stack, int curr_index, global char* curr_address, float3 corner_far, float3 voxelCentre, float t_min, float t_max) {
	if(curr_index >= STACK_SIZE) {
		curr_index = STACK_SIZE - 1;
		for(int i = 1; i < STACK_SIZE; i++) 
			short_stack[i-1]=short_stack[i];
	}
	
	short_stack[curr_index].address = curr_address;
	short_stack[curr_index].far_corner = corner_far;
	short_stack[curr_index].node_centre = voxelCentre;
	short_stack[curr_index].t_min = t_min;
	short_stack[curr_index].t_max = t_max;
	curr_index++;
	return curr_index;
}

global char* find_collission(global char* octree, float3 origin, float3 direction, float t) {

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
			
	global char* curr_address = octree;
	float3 voxelCentre = (float3)(0.0f);
	bool collission = false;	
	int curr_index = 0;
	struct stack short_stack[STACK_SIZE];
			
	/* We are out of the volume and we will never get to it. */
	if(t >= t_max) {
		collission = true;
		curr_address = 0; 
	}

	while(!collission) {

		if(t >= t_out) {
			collission = true;
			curr_address = 0;
		} else if(no_children(curr_address)) {
            collission = true;
        } else {
            /*If we are inside the node*/
            if(t_min <= t && t < t_max) {
                float3 rayPos = origin + (direction * t);
                
                char xyz_flag = makeXYZFlag(rayPos, voxelCentre);
                float nodeHalfSize = fabs((corner_far-voxelCentre).x)/2.0f;
                
                float3 tmpNodeCentre = (float3)( xyz_flag & 1 ? voxelCentre.x + nodeHalfSize : voxelCentre.x - nodeHalfSize,
												 xyz_flag & 2 ? voxelCentre.y + nodeHalfSize : voxelCentre.y - nodeHalfSize,
												 xyz_flag & 4 ? voxelCentre.z + nodeHalfSize : voxelCentre.z - nodeHalfSize);
                
                float3 tmp_corner_far = (float3)(direction.x >= 0 ? tmpNodeCentre.x + nodeHalfSize : tmpNodeCentre.x - nodeHalfSize,
												 direction.y >= 0 ? tmpNodeCentre.y + nodeHalfSize : tmpNodeCentre.y - nodeHalfSize,
												 direction.z >= 0 ? tmpNodeCentre.z + nodeHalfSize : tmpNodeCentre.z - nodeHalfSize);
                
                float tmp_max = min_component((tmp_corner_far - origin) / direction);
                
                if(nodeHasChildAt(rayPos, voxelCentre, curr_address)) {
                    /* If the voxel we are at is not empty, go down. */
                    
                    curr_index = push(short_stack, curr_index, curr_address, corner_far, voxelCentre, t_min, t_max);
                    
                    curr_address = getChild(rayPos,voxelCentre,curr_address);
                    
                    corner_far = tmp_corner_far;
                    voxelCentre = tmpNodeCentre;
                    corner_close =  (float3)(direction.x >= 0 ? voxelCentre.x - nodeHalfSize : voxelCentre.x + nodeHalfSize,
                                             direction.y >= 0 ? voxelCentre.y - nodeHalfSize : voxelCentre.y + nodeHalfSize,
                                             direction.z >= 0 ? voxelCentre.z - nodeHalfSize : voxelCentre.z + nodeHalfSize);
                    t_max = tmp_max;
                    t_min = max_component((corner_close - origin) / direction);
                    
                } else {
                    /* If the child is empty, we step the ray. */
                    t = tmp_max;
                }
            } else {
                /* We are outside the node. Pop the stack */
                curr_index--;
                if(curr_index>=0) {
                    /* Pop that stack! */
                    corner_far = short_stack[curr_index].far_corner;
					voxelCentre = short_stack[curr_index].node_centre;
					curr_address = short_stack[curr_index].address;
					t_min = short_stack[curr_index].t_min;
					t_max = short_stack[curr_index].t_max;
                } else {
                    /* Since we are using a short stack, we restart from the root node. */
					curr_index = 0;
                    curr_address = octree;
                    collission = true;
					corner_far = (float3)(direction.x >= 0 ? half_size : -half_size,
										  direction.y >= 0 ? half_size : -half_size,
										  direction.z >= 0 ? half_size : -half_size);

					corner_close = (float3)(-corner_far.x,-corner_far.y,-corner_far.z);

					t_min = max_component((corner_close - origin) / direction); 
					t_max = min_component((corner_far - origin) / direction);
                }
            }
        }
	}

	return curr_address;
}

kernel void ray_trace(global char* octree,
                      struct renderinfo info,
                      int widthOfFramebuffer,
                        global unsigned char* frameBuff) {
    int x = get_global_id(0);
	int y = get_global_id(1);

	float3 o = info.viewPortStart + (info.viewStep * x) + (info.up * y);
    float3 d = o-info.eyePos; 

	global char* voxel = find_collission(octree, o, d, 1.0f);

	if(voxel) {
		global char* attr = get_attributes(voxel);

		int pixel_index = (x*3)+(y*widthOfFramebuffer*3);
		frameBuff[pixel_index + 0] = attr[0];
		frameBuff[pixel_index + 1] = attr[1];
		frameBuff[pixel_index + 2] = attr[2];
	}
}