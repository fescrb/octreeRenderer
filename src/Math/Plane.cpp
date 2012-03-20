#include "Plane.h"

float4 plane::getIntersectionPoint(const line& intersecting_line) {
    float4 origin = intersecting_line.getOrigin();
    float4 direction = intersecting_line.getDirection();
    
    float scalar = ( dot(m_normal,origin) - dot(m_normal, m_pointInPlane) ) / dot(m_normal, direction);
    
    return intersecting_line.getPositionAt(scalar);
}

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
        
        vertices.clear();
        missing_vertices.clear();
        
        // Count how many are outside the plane.
        int counter = 0;
        for(int i = 0; i < 3; i++)
            if(within_pane[i]) {
                counter++;
                vertices.push_back(triangleToCull.getVertex(i));
            } else
                missing_vertices.push_back(triangleToCull.getVertex(i));
        
        if(counter == 1) {
            vertex new_vertex_one = triangleToCull.interpolate(getIntersectionPoint(line(vertices[0].getPosition(), 
                                                                                         missing_vertices[0].getPosition() - vertices[0].getPosition())));
            vertex new_vertex_two = triangleToCull.interpolate(getIntersectionPoint(line(vertices[0].getPosition(),
                                                                                         missing_vertices[1].getPosition() - vertices[0].getPosition())));
            
            float4 normal = cross(new_vertex_one.getPosition()-vertices[0].getPosition(),new_vertex_two.getPosition()-vertices[0].getPosition());
            
            if(dot(vertices[0].getNormal(),normal) > 0) {
                result.push_back(triangle(vertices[0],new_vertex_two,new_vertex_one));
            } else {
                result.push_back(triangle(vertices[0],new_vertex_two,new_vertex_one));
            }
        } else {
             vertex new_vertex_one = triangleToCull.interpolate(getIntersectionPoint(line(vertices[0].getPosition(), 
                                                                                          missing_vertices[0].getPosition() - vertices[0].getPosition())));
             vertex new_vertex_two = triangleToCull.interpolate(getIntersectionPoint(line(vertices[1].getPosition(),
                                                                                          missing_vertices[0].getPosition() - vertices[1].getPosition())));
             
             float4 normal = cross(new_vertex_one.getPosition()-vertices[0].getPosition(), vertices[1].getPosition()-vertices[0].getPosition());
             
             if(dot(vertices[0].getNormal(),normal) > 0) {
                result.push_back(triangle(vertices[0],vertices[1],new_vertex_one));
                result.push_back(triangle(vertices[1],new_vertex_two,new_vertex_one));
            } else {
                result.push_back(triangle(vertices[0],new_vertex_one,vertices[1]));
                result.push_back(triangle(vertices[1],new_vertex_one,new_vertex_two));
            }
        }
    }
    
    return result;
}