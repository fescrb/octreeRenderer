#include "Plane.h"


std::vector<triangle> plane::cull(const triangle& triangleToCull) {
    std::vector<triangle> result(0);
    
    bool within_pane[3] = { 
        insidePlane(triangleToCull.getVertex(0).getPosition()),
        insidePlane(triangleToCull.getVertex(1).getPosition()),
        insidePlane(triangleToCull.getVertex(2).getPosition())
    };
    
    if(within_pane[0] && within_pane[1] && within_pane[2]) {
        result.push_back(triangleToCull);
    } else if (within_pane[0] || within_pane[1] || within_pane[2]) {
        std::vector<vertex> vertices;
        std::vector<vertex> missing_vertices;
        
        vertices.push_back();
        
        // Count how many are outside the plane.
        int counter = 0;
        for(int i = 0; i < 3; i++)
            if(within_pane[i]) {
                counter++;
                vertices.push_back(triangleToCull.getVertex(i));
            } else
                missing_vertices.push_back(triangleToCull.getVertex(i));
        
        if(counter == 1) {
                
        } else {
            
        }
    }
    
    return result;
}