#include "Mesh.h"

#include "Graphics.h"

std::vector<vertex> mesh::getOuterMostVertices() const {
    // We initialise the list of vertices with the first vertex of the mesh;
    std::vector<vertex> vertices(6, m_triangles[0].getVertex(0));
    
    for(int i = 0; i < vertices.size(); i++) 
        for(int j = 0; j < 3; j++) {
            vertex this_vertex = m_triangles[i].getVertex(j);
            float4 vertex_pos = this_vertex.getPosition();
            if(vertex_pos.getX() > vertices[0].getPosition().getX())
                vertices[0] = this_vertex;
            if(vertex_pos.getX() < vertices[1].getPosition().getX())
                vertices[1] = this_vertex;
            if(vertex_pos.getY() > vertices[2].getPosition().getY())
                vertices[2] = this_vertex;
            if(vertex_pos.getY() < vertices[3].getPosition().getY())
                vertices[3] = this_vertex;
            if(vertex_pos.getZ() > vertices[4].getPosition().getZ())
                vertices[4] = this_vertex;
            if(vertex_pos.getZ() < vertices[5].getPosition().getZ())
                vertices[5] = this_vertex;
        }
       
    return vertices;
}

void mesh::render() const {
    glBegin(GL_TRIANGLES);
    
    for(int i = 0; i < m_triangles.size(); i++) 
        for(int j = 0; j < 3; j++) {
            vertex this_vertex = m_triangles[i].getVertex(j);
            float4 colour = this_vertex.getColour();
            float4 normal = this_vertex.getNormal();
            float4 position = this_vertex.getPosition();
            glColor4f(colour.getX(), colour.getY(), colour.getZ(), colour.getW());
            glNormal3f(normal.getX(), normal.getY(), normal.getZ());
            glVertex4f(position.getX(), position.getY(), position.getZ(), position.getW());
        }

    glEnd();
}