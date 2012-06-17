#define STACK_SIZE 12

#define F32_EPSILON 1E-6
#define OCTREE_ROOT_HALF_SIZE 1.0f

#define WINDOW_SIZE 32
#define WINDOW_PIXEL_COUNT 1024
#define RAY_BUNDLE_WINDOW_SIZE 4

struct renderinfo{
	float3 eyePos, viewDir, up, viewPortStart, viewStep;
	float eyePlaneDist, fov, pixel_half_size;

    float3 lightPos;
    float lightBrightness;
};

struct collission {
    global char* node_pointer;
    float t;
    unsigned short iterations;
    unsigned char depth_in_octree;
};

struct stack{
	global char* address;
	float3 far_corner, node_centre, corner_close;
	float t_max;
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
    return !address[3];
}

global char* get_attributes(global char* node) {
    return node + ((((global unsigned char*)node)[2] >> 4) * 4);
}

uchar3 getColours(global char* attr) {
    unsigned short rgb_565 = ((global unsigned short*)attr)[0];
    uchar3 colour;
    
    colour.x = (rgb_565 >> 8) & ~7;
    colour.y = ((unsigned char)(rgb_565 >> 3)) & ~3;
    colour.z = (rgb_565 & 31) << 3;
    
    return colour;
}

float4 getNormal(global char* attr) {
    unsigned short normals_short = ((global unsigned short*)attr)[1];
    float4 normal;
    
    char x = normals_short >> 8;
    normal.x = fixed_point_8bit_to_float(x);
    char y = 0 | (normals_short & 254);
    normal.y = fixed_point_8bit_to_float(y);
    
    float z = sqrt(1 - (normal.x*normal.x) - (normal.y*normal.y));
    if(normals_short & 1)
        z*=-1.0f;
    
    normal.z = z;
    normal.w = 0.0f;
    
    return normal;
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
	return node[3] & (1 << xyz_flag);  
}

global char* getChild(global char* node, char xyz_flag) {
    global int *node_int = (global int*)node;
    int pos = 0;//(node_int[0] >> (xyz_flag * 3)) & 0b111;
    switch(xyz_flag) {
    case 1:
        pos = node_int[0] & 1;
        break;
    case 2:
        pos = (node_int[0] >> 1) & 3;
        break;
    case 3:
        pos = (node_int[0] >> 3) & 3;
        break;
    case 4:
        pos = (node_int[0] >> 5) & 7;
        break;
    case 5:
        pos = (node_int[0] >> 8) & 7;
        break;
    case 6:
        pos = (node_int[0] >> 11) & 7;
        break;
    case 7:
        pos = (node_int[0] >> 14) & 7;
        break;
    default:
        break;
    }
    int diff = 0;
    if(node[2] & 2) {
        global unsigned short *node_short = (global unsigned short*)node;
        node_short+=(pos+2);
        diff = node_short[0];
        pos /= 2;
    } else {
        node_int+=(pos+1);
        diff = node_int[0];
    }
    node+=(pos+1)*4;
    return node + (diff*4);
}

int push(struct stack* short_stack, int curr_index, global char* curr_address, float3 corner_far, float3 voxelCentre, float3 corner_close, float t_max) {
	if(curr_index >= STACK_SIZE) {
		curr_index = STACK_SIZE - 1;
		for(int i = 1; i < STACK_SIZE; i++) 
			short_stack[i-1]=short_stack[i];
	}
	
	short_stack[curr_index].address = curr_address;
	short_stack[curr_index].far_corner = corner_far;
	short_stack[curr_index].node_centre = voxelCentre;
	short_stack[curr_index].t_max = t_max;
    short_stack[curr_index].corner_close = corner_close;
	curr_index++;
	return curr_index;
}

struct collission find_collission(global char* octree, float3 origin, float3 direction, float t, float pixel_half_size) {
    unsigned short it = 1;
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
	private struct stack short_stack[STACK_SIZE];
			
	/* We are out of the volume and we will never get to it. */
	if(t >= t_max) {
		collission = true;
		curr_address = 0; 
	}

	while(!collission) {
        it++;
		if(t >= t_out) {
			collission = true;
			curr_address = 0;
		} else if(no_children(curr_address)) {
            collission = true;
        } else {
            /*If we are inside the node*/
            if(t < t_max) {
                float3 t_centre_vector = (voxelCentre - origin) / direction;

                char xyz_flag = makeXYZFlag(t_centre_vector, t, direction);
                float nodeHalfSize = half_size/2.0f;
                
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
                    if(nodeHalfSize < pixel_half_size*t) {
                        collission = true;
                        break;
                    }
                    
                    curr_index = push(short_stack, curr_index, curr_address, corner_far, voxelCentre, corner_close, t_max);
                    
                    curr_address = getChild(curr_address, xyz_flag);
                    voxelCentre = tmpNodeCentre;
                    t_max = tmp_max;

                    depth_in_octree++;
                    half_size = nodeHalfSize;
                    
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
					t_max = short_stack[curr_index].t_max;
                    corner_close = short_stack[curr_index].corner_close;
                    half_size*=2;
                    depth_in_octree--;
                } else {
                    /* Since we are using a short stack, we restart from the root node. */
					curr_index = 0;
                    curr_address = octree;
                    half_size = OCTREE_ROOT_HALF_SIZE;
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

kernel void calculate_costs(read_only image2d_t itBuff, write_only global uint* costs) {
    local uint local_costs[RAY_BUNDLE_WINDOW_SIZE];

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    int x = get_global_id(0);
    const int height = get_image_height(itBuff);

    uint val = 0;
    for(int y = 0; y < height; y++) {
        val+= (read_imagef(itBuff, sampler, (int2)(x, y))).x*512.0f;
    }

    local_costs[get_local_id(0)] = val;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0)==0) {
        for(int i = 1; i < RAY_BUNDLE_WINDOW_SIZE; i++) {
            val+= local_costs[i];
        }
        costs[x/RAY_BUNDLE_WINDOW_SIZE] = val;
    }
}

kernel void clear_buffer(write_only image2d_t buffer) {
    write_imagef(buffer, (int2)(get_global_id(0), get_global_id(1)), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
}

kernel void clear_uintbuffer(write_only global uint* costs) {
    costs[get_global_id(0)] = 0;
}

kernel void ray_trace(global char* octree,
                      global char* header,
                      struct renderinfo info,
                      write_only image2d_t frameBuff,
                      read_only  image2d_t depthBuff,
                      write_only image2d_t itBuff) {
    int x = get_global_id(0);
	int y = get_global_id(1);

	float3 o = info.viewPortStart + (info.viewStep * x) + (info.up * y);
    float3 d = o-info.eyePos;
    o = info.eyePos;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
    float4 t_start = read_imagef(depthBuff, sampler, (int2)(x,y));

	struct collission col = find_collission(octree, o, d, t_start.x, info.pixel_half_size);

    float3 rayPos = o + (d * col.t);

    float ambient = 0.2f;

	if(col.node_pointer) {
		global char* attr = get_attributes(col.node_pointer);
        
        uchar3 colour = getColours(attr);

        unsigned char red = colour.x;
        unsigned char green = colour.y;
        unsigned char blue = colour.z;
        
        //Fixed direction light coming from (1, 1, 1);
        float4 direction_towards_light = normalize((float4)(info.lightPos - rayPos,0.0f));
        float4 normal = getNormal(attr);
        // K_diff is always 1, for now
        float diffuse_coefficient = dot(direction_towards_light,normal);
        if(diffuse_coefficient<0)
            diffuse_coefficient=0.0f;
        red=(red*diffuse_coefficient*(1.0f-ambient))+(red*ambient);
        green=(green*diffuse_coefficient*(1.0f-ambient))+(green*ambient);
        blue=(blue*diffuse_coefficient*(1.0f-ambient))+(blue*ambient);

        if(get_image_channel_data_type(frameBuff) == CLK_UNSIGNED_INT8) {
            uint4 color = (uint4)(red, green, blue, 255);
            //uint4 color = (uint4)(col.iterations, col.iterations, col.iterations, 255);
            write_imageui ( frameBuff, (int2)(x, y), color);
        } else {
            float4 color = (float4)(red/255.0f, green/255.0f, blue/255.0f, 1.0f);
            //float4 color = (float4)(col.iterations/255.0f, col.iterations/255.0f, col.iterations/255.0f, 1.0f);
            //char color_per_level = 255/(((global int*)header)[0] - 1);
            //uint4 color = (uint4)(col.depth_in_octree*color_per_level, col.depth_in_octree*color_per_level, col.depth_in_octree*color_per_level, 255);
            //float dep = fabs(dot(rayPos, info.viewDir))/(OCTREE_ROOT_HALF_SIZE*2.0f);
            //uint4 color = (uint4)(255*dep, 255*dep, 255*dep, 255);

            write_imagef ( frameBuff, (int2)(x, y), color);
        }
    }
    write_imagef(itBuff, (int2)(x, y), (float4)(col.iterations/512.0f)); 
}

kernel void trace_bundle(global char* octree,
                         global char* header,
                         struct renderinfo info,
                         int width,
                         write_only  image2d_t depthBuff) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float3 origin = info.eyePos;
    float3 from_centre_to_start = -(info.viewStep/2.0f) - (info.up/2.0f);
    float3 d_lower_left = ((info.viewPortStart + (info.viewStep * (x*width)) + (info.up * (y*width)))-origin);
    d_lower_left+=from_centre_to_start;
    float3 d_upper_left = ((info.viewPortStart + (info.viewStep * (x*width)) + (info.up * ((y+1)*width)))-origin);
    d_upper_left+=from_centre_to_start;
    float3 d_lower_right = ((info.viewPortStart + (info.viewStep * ((x+1)*width)) + (info.up * (y*width)))-origin);
    d_lower_right+=from_centre_to_start;
    float3 d_upper_right = ((info.viewPortStart + (info.viewStep * ((x+1)*width)) + (info.up * ((y+1)*width)))-origin);
    d_upper_right+=from_centre_to_start;
    
    float3 direction = d_lower_left;
    
    float t = 0.0f;
    float t_prev = t;

    float half_size = OCTREE_ROOT_HALF_SIZE;

    float pixel_half_size = info.pixel_half_size * width;

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
            if(t < t_max) {
                 // We check if all rays fit
                if(max_component((corner_close - origin)/d_lower_right) >= min_component((corner_far - origin) / d_lower_right)) 
                    collission = true;
                
                if(max_component((corner_close - origin)/d_upper_left) >= min_component((corner_far - origin) / d_upper_left)) 
                    collission = true;
                
                if(max_component((corner_close - origin)/d_upper_right) >= min_component((corner_far - origin) / d_upper_right)) 
                    collission = true;
                
                if(collission) {
                    //t = t_prev;
                    break;
                }
                
                float3 t_centre_vector = (voxelCentre - origin) / direction;

                char xyz_flag = makeXYZFlag(t_centre_vector, t, direction);
                float nodeHalfSize = half_size/2.0f;
                
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
                    if(nodeHalfSize < pixel_half_size*t) {
                        collission = true;
                        break;
                    }
                    
                    curr_index = push(short_stack, curr_index, curr_address, corner_far, voxelCentre, corner_close, t_max);
                    
                    curr_address = getChild(curr_address, xyz_flag);
                    
                    corner_far = tmp_corner_far;
                    voxelCentre = tmpNodeCentre;
                    corner_close =  (float3)(direction.x >= 0 ? voxelCentre.x - nodeHalfSize : voxelCentre.x + nodeHalfSize,
                                             direction.y >= 0 ? voxelCentre.y - nodeHalfSize : voxelCentre.y + nodeHalfSize,
                                             direction.z >= 0 ? voxelCentre.z - nodeHalfSize : voxelCentre.z + nodeHalfSize);
                    t_max = tmp_max;
                    half_size = nodeHalfSize;
                    
                } else {
                    /* If the child is empty, we step the ray. */
                    t_prev = t;
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
                    t_max = short_stack[curr_index].t_max;
                    corner_close = short_stack[curr_index].corner_close;
                    half_size*=2;
                } else {
                    /* Since we are using a short stack, we restart from the root node. */
                    curr_index = 0;
                    curr_address = octree;
                    half_size = OCTREE_ROOT_HALF_SIZE;
                    //collission = true;
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
    for(x = get_global_id(0)*width; x < (get_global_id(0)+1)*width; x++) {
        for(y = get_global_id(1)*width; y < (get_global_id(1)+1)*width; y++) {
            write_imagef ( depthBuff, (int2)(x, y), (float4)(t_prev));
        }
    }
}