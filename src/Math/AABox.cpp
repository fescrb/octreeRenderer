#include "AABox.h"

#include "Plane.h"

#include <cstdio>

aabox::aabox(const mesh& meshToBound) {
    std::vector<vertex> max_min = meshToBound.getOuterMostVertices();
    float x_max = max_min[0].getPosition().getX();
    float x_min = max_min[1].getPosition().getX();
    float y_max = max_min[2].getPosition().getY();
    float y_min = max_min[3].getPosition().getY();
    float z_max = max_min[4].getPosition().getZ();
    float z_min = max_min[5].getPosition().getZ();
    
    m_corner = position(float3(x_min, y_min, z_min));
    m_sizes = float3(x_max-x_min, y_max-y_min, z_max-z_min);
}

mesh aabox::cull(const mesh& meshToCull) {
    mesh resultantMesh;
    
    plane aabox_planes[6] = {
        plane(getCorner(), direction(float3(getSizes()[0] > 0 ? 1.0f : -1.0f, 0.0f, 0.0f))),
        plane(getCorner(), direction(float3(0.0f, getSizes()[0] > 0 ? 1.0f : -1.0f, 0.0f))),
        plane(getCorner(), direction(float3(0.0f, 0.0f, getSizes()[0] > 0 ? 1.0f : -1.0f))),
        plane(getFarCorner(), direction(float3(getSizes()[0] < 0 ? 1.0f : -1.0f, 0.0f, 0.0f))),
        plane(getFarCorner(), direction(float3(0.0f, getSizes()[0] < 0 ? 1.0f : -1.0f, 0.0f))),
        plane(getFarCorner(), direction(float3(0.0f, 0.0f, getSizes()[0] < 0 ? 1.0f : -1.0f))),
    };
    
    int mesh_size = meshToCull.getTriangleCount();
    
    std::vector<triangle> to_process; to_process.clear();
    std::vector<triangle> to_add; to_add.clear();
    
    for(int i = 0; i < mesh_size; i++) {
        to_process.clear();
        to_process.push_back(meshToCull.getTriangle(i));
        
        printf("triangle to process: (%f %f %f) (%f %f %f) (%f %f %f)\n"
            , to_process[0].getVertex(0).getPosition()[0]
            , to_process[0].getVertex(0).getPosition()[1]
            , to_process[0].getVertex(0).getPosition()[2]
            , to_process[0].getVertex(1).getPosition()[0]
            , to_process[0].getVertex(1).getPosition()[1]
            , to_process[0].getVertex(1).getPosition()[2]
            , to_process[0].getVertex(2).getPosition()[0]
            , to_process[0].getVertex(2).getPosition()[1]
            , to_process[0].getVertex(2).getPosition()[2]
        );
        
        for(int j = 0; j < 6 && !to_process.empty(); j++) {
            to_add.clear();
            
            while(!to_process.empty()) {
                triangle this_triangle = to_process[to_process.size() - 1];
                to_process.pop_back();
                
                printf("this triange: (%f %f %f) (%f %f %f) (%f %f %f)\n"
                    , this_triangle.getVertex(0).getPosition()[0]
                    , this_triangle.getVertex(0).getPosition()[1]
                    , this_triangle.getVertex(0).getPosition()[2]
                    , this_triangle.getVertex(1).getPosition()[0]
                    , this_triangle.getVertex(1).getPosition()[1]
                    , this_triangle.getVertex(1).getPosition()[2]
                    , this_triangle.getVertex(2).getPosition()[0]
                    , this_triangle.getVertex(2).getPosition()[1]
                    , this_triangle.getVertex(2).getPosition()[2]
                );
                
                printf("this plane (%f %f %f) (%f %f %f)\n"
                    , aabox_planes[j].getPointInPlane()[0]
                    , aabox_planes[j].getPointInPlane()[1]
                    , aabox_planes[j].getPointInPlane()[2]
                    , aabox_planes[j].getNormal()[0]
                    , aabox_planes[j].getNormal()[1]
                    , aabox_planes[j].getNormal()[2]
                );
                
                std::vector<triangle> this_result = aabox_planes[j].cull(this_triangle);
                
                for(int k = 0; k < this_result.size(); k++) {
                    printf("this result %d: (%f %f %f) (%f %f %f) (%f %f %f)\n", k
                        , this_result[k].getVertex(0).getPosition()[0]
                        , this_result[k].getVertex(0).getPosition()[1]
                        , this_result[k].getVertex(0).getPosition()[2]
                        , this_result[k].getVertex(1).getPosition()[0]
                        , this_result[k].getVertex(1).getPosition()[1]
                        , this_result[k].getVertex(1).getPosition()[2]
                        , this_result[k].getVertex(2).getPosition()[0]
                        , this_result[k].getVertex(2).getPosition()[1]
                        , this_result[k].getVertex(2).getPosition()[2]
                    );
                }
                
                printf("--------------\n");
                
                to_add.insert(to_add.end(), this_result.begin(), this_result.end());
            }
            
            to_process = to_add;
        }
        
        // We check if any triangle is a single point or a line.
        for(int k = 0; k < to_add.size(); k++) {
            
            if(    to_add[k].getVertex(0).getPosition() != to_add[k].getVertex(1).getPosition()
                && to_add[k].getVertex(0).getPosition() != to_add[k].getVertex(2).getPosition()
                && to_add[k].getVertex(1).getPosition() != to_add[k].getVertex(2).getPosition()
            )
                resultantMesh.appendTriangle(to_add[k]);
        }
    }

    return resultantMesh;
}
