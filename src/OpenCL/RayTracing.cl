#define STACK_SIZE 10

#define F32_EPSILON 1E-6
#define OCTREE_ROOT_HALF_SIZE 1.0f

struct renderinfo{
	float3 eyePos, viewDir, up, viewPortStart, viewStep;
	float eyePlaneDist, fov, pixel_half_size;

    float3 lightPos;
    float lightBrightness;
};

struct collission {
    global char* node_pointer;
    float t;
    unsigned char iterations;
    unsigned char depth_in_octree;
};

struct stack{
	global char* address;
	float3 far_corner, node_centre, corner_close;
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

float fixed_point_8bit_to_float(char fixed) {
    float range =127.0f; // Max value of a 7 bit unsigned integer.
    float step = 1.0f/range;
    return fixed*step;
}

bool no_children(__global char* address) {
    return !address[7];
}

char makeXYZFlag(float3 t_centre_vector, float t, float3 direction) {
    char flag = 0;

    if( t >= t_centre_vector.x ) {
        if(direction.x >= 0.0f)
            flag |= 1; 
    } else {
        if(direction.x < 0.0f)
            flag |= 1; 
    }

    if( t >= t_centre_vector.y ) {
        if(direction.y >= 0.0f)
            flag |= 2; 
    } else {
        if(direction.y < 0.0f)
            flag |= 2; 
    }

    if( t >= t_centre_vector.z ) {
        if(direction.z >= 0.0f)
            flag |= 4; 
    } else {
        if(direction.z < 0.0f)
            flag |= 4; 
    }

    return flag;
}

bool nodeHasChildAt(global char* node, char xyz_flag) {
	return node[7] & (1 << xyz_flag);  
}

global char* getChild(global char* node, char xyz_flag) {
    global int *node_int = (global int*)node;
    int pos = (node_int[0] >> (xyz_flag * 3)) & 0b111;
    node_int+=(pos+2);
    node+=(pos+2)*4;
    return node + (node_int[0]*4);
}

global char* get_attributes(global char* node) {
    global int* addr_int = (global int*)node;
    return node + ((addr_int[1] & ~(255 << 24)) * 4) +4;
}

int push(struct stack* short_stack, int curr_index, global char* curr_address, float3 corner_far, float3 voxelCentre, float3 corner_close, float t_min, float t_max) {
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
    short_stack[curr_index].corner_close = corner_close;
	curr_index++;
	return curr_index;
}

struct collission find_collission(global char* octree, float3 origin, float3 direction, float t, float pixel_half_size) {
    unsigned char it = 0;
    unsigned char depth_in_octree = 0;

	float half_size = OCTREE_ROOT_HALF_SIZE;

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

    float3 rayPos = origin;

	while(!collission) {
        it++;
		if(t >= t_out) {
			collission = true;
			curr_address = 0;
		} else if(no_children(curr_address)) {
            collission = true;
        } else {
            /*If we are inside the node*/
            if(t_min <= t && t < t_max) {
                rayPos = origin + (direction * t);
                
                float3 t_centre_vector = (voxelCentre - origin) / direction;

                char xyz_flag = makeXYZFlag(t_centre_vector, t, direction);
                float nodeHalfSize = fabs((corner_far-voxelCentre).x)/2.0f;
                
                float3 tmpNodeCentre = (float3)( xyz_flag & 1 ? voxelCentre.x + nodeHalfSize : voxelCentre.x - nodeHalfSize,
												 xyz_flag & 2 ? voxelCentre.y + nodeHalfSize : voxelCentre.y - nodeHalfSize,
												 xyz_flag & 4 ? voxelCentre.z + nodeHalfSize : voxelCentre.z - nodeHalfSize);
                
                float3 tmp_corner_far = (float3)(direction.x >= 0 ? tmpNodeCentre.x + nodeHalfSize : tmpNodeCentre.x - nodeHalfSize,
												 direction.y >= 0 ? tmpNodeCentre.y + nodeHalfSize : tmpNodeCentre.y - nodeHalfSize,
												 direction.z >= 0 ? tmpNodeCentre.z + nodeHalfSize : tmpNodeCentre.z - nodeHalfSize);
                
                float tmp_max = min_component((tmp_corner_far - origin) / direction);
                
                if(nodeHasChildAt(curr_address, xyz_flag)) {
                    /* If the voxel we are at is not empty, go down. */

                    // We check for LOD.
                    if(nodeHalfSize*2.0f < pixel_half_size*t) {
                        collission = true;
                        break;
                    }
                    
                    curr_index = push(short_stack, curr_index, curr_address, corner_far, voxelCentre, corner_close, t_min, t_max);
                    
                    curr_address = getChild(curr_address, xyz_flag);
                    
                    corner_far = tmp_corner_far;
                    voxelCentre = tmpNodeCentre;
                    corner_close =  (float3)(direction.x >= 0 ? voxelCentre.x - nodeHalfSize : voxelCentre.x + nodeHalfSize,
                                             direction.y >= 0 ? voxelCentre.y - nodeHalfSize : voxelCentre.y + nodeHalfSize,
                                             direction.z >= 0 ? voxelCentre.z - nodeHalfSize : voxelCentre.z + nodeHalfSize);
                    t_max = tmp_max;
                    /*float t_min_x = (corner_close.x - origin.x) / direction.x;
                    float t_min_y = (corner_close.y - origin.y) / direction.y;
                    float t_min_z = (corner_close.z - origin.z) / direction.z;
                    t_min = t_min_x > t_min_y? t_min_x : t_min_y;
                    t_min = t_min > t_min_z? t_min : t_min_z;*/
                    t_min = max_component((corner_close - origin) / direction);

                    depth_in_octree++;
                    
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
                    corner_close = short_stack[curr_index].corner_close;
                    depth_in_octree--;
                } else {
                    /* Since we are using a short stack, we restart from the root node. */
					curr_index = 0;
                    curr_address = octree;
                    //collission = true;
					corner_far = (float3)(direction.x >= 0 ? half_size : -half_size,
										  direction.y >= 0 ? half_size : -half_size,
										  direction.z >= 0 ? half_size : -half_size);

					corner_close = (float3)(-corner_far.x,-corner_far.y,-corner_far.z);

					t_min = max_component((corner_close - origin) / direction); 
					t_max = min_component((corner_far - origin) / direction);
                    depth_in_octree = 0;
                }
            }
        }
	}

    //if(it == 255)
    //    curr_address = 0;

    struct collission col;
    col.node_pointer = curr_address;
    col.t = t;
    col.iterations = it;
    col.depth_in_octree = depth_in_octree;

	return col;
}

kernel void clear_framebuffer(write_only image2d_t frameBuff) {
    write_imageui ( frameBuff, (int2)(get_global_id(0), get_global_id(1)), (uint4)(0, 0, 0, 0));
}

kernel void ray_trace(global char* octree,
                      global char* header,
                      struct renderinfo info,
                      int2 origin,
                      write_only image2d_t frameBuff,
                      write_only image2d_t depthBuff) {
    int x = origin.x + get_global_id(0);
	int y = origin.y + get_global_id(1);

	float3 o = info.viewPortStart + (info.viewStep * x) + (info.up * y);
    float3 d = o-info.eyePos;
    o = info.eyePos;

	struct collission col = find_collission(octree, o, d, 1.0f, info.pixel_half_size);

    float3 rayPos = o + (d * col.t);

    float ambient = 0.2f;

	if(col.node_pointer) {
		global char* attr = get_attributes(col.node_pointer);

        unsigned char red = attr[0];
        unsigned char green = attr[1];
        unsigned char blue = attr[2];

        // If attributes contains a normal
        if(((global int*)header)[1] > 4) {
            //Fixed direction light coming from (1, 1, 1);
            float4 direction_towards_light = normalize((float4)(info.lightPos - rayPos,0.0f));
            float4 normal = (float4)(fixed_point_8bit_to_float(attr[4]),
                                     fixed_point_8bit_to_float(attr[5]),
                                     fixed_point_8bit_to_float(attr[6]),
                                     fixed_point_8bit_to_float(attr[7]));
            // K_diff is always 1, for now
            float diffuse_coefficient = dot(direction_towards_light,normal);
            if(diffuse_coefficient<0)
                diffuse_coefficient=0.0f;
            red=(red*diffuse_coefficient*(1.0f-ambient))+(red*ambient);
            green=(green*diffuse_coefficient*(1.0f-ambient))+(green*ambient);
            blue=(blue*diffuse_coefficient*(1.0f-ambient))+(blue*ambient);
        }

        uint4 color = (uint4)(red, green, blue, 255);
        //uint4 color = (uint4)(col.iterations, col.iterations, col.iterations, 255);
        //char color_per_level = 255/((global int*)header)[0];
        //uint4 color = (uint4)(col.depth_in_octree*color_per_level, col.depth_in_octree*color_per_level, col.depth_in_octree*color_per_level, 255);

        write_imageui ( frameBuff, (int2)(x, y), color);
    }
    write_imagef ( depthBuff, (int2)(x, y), (float4)(/*(float)col.depth_in_octree*/col.t));
}