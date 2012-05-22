#include "OctreeCreator.h"

#include "AABox.h"

#include "Matrix.h"

#include <cstdio>

OctreeCreator::OctreeCreator(mesh meshToConvert, int depth)
:   m_mesh(meshToConvert),
    m_aabox(meshToConvert),
    m_depth(depth){
    // We need to centre the mesh at the origin.
    float4 off_centre_difference = float4() - m_aabox.getCentre();

    float4x4 translation_matrix = float4x4::translationMatrix(off_centre_difference[0], off_centre_difference[1], off_centre_difference[2]);
    
    m_mesh = translation_matrix * m_mesh;
    m_aabox = translation_matrix * m_aabox;
}

void OctreeCreator::render() {
    m_mesh.render();
}

void OctreeCreator::convert() {
    octree<mesh*> mesh_octree = octree<mesh*>();
    float3 half_size = m_aabox.getSizes()/2.0f;
    float4 centre = m_aabox.getCentre();
    float4 corner = m_aabox.getCorner();
    
    mesh_octree.m_node = new mesh(m_mesh);
    mesh_octree.allocateChildren();
    
    for(int i = 0; i < 8; i++) {
        float4 new_corner = float4(0.0f, 0.0f, 0.0f, 1.0f);
        
        i & octree<mesh>::X ? new_corner.setX(centre[0]): new_corner.setX(corner[0]);
        i & octree<mesh>::Y ? new_corner.setY(centre[1]): new_corner.setY(corner[1]);
        i & octree<mesh>::Z ? new_corner.setZ(centre[2]): new_corner.setZ(corner[2]);
        
        if( createSubtree(&(mesh_octree.m_children[i]), *(mesh_octree.m_node), aabox(new_corner, half_size), m_depth)) {
            mesh_octree.m_children_flag |= (1<<i);
        }
    }
    
    /*for(int i = 0; i < res_mesh.getTriangleCount(); i++) {
        triangle tri = res_mesh.getTriangle(i);
        
        vertex v = tri.getVertex(0); float4 pos = v.getPosition(); float4 nor = v.getNormal();
        printf("Triangle %d vertex 1: x %f y %f z %f nx %f ny %f nz %f\n", i, pos[0], pos[1], pos[2], nor[0], nor[1], nor[2]);
        v = tri.getVertex(1); pos = v.getPosition(); nor = v.getNormal();
        printf("Triangle %d vertex 2: x %f y %f z %f nx %f ny %f nz %f\n", i, pos[0], pos[1], pos[2], nor[0], nor[1], nor[2]);
        v = tri.getVertex(2); pos = v.getPosition(); nor = v.getNormal();
        printf("Triangle %d vertex 3: x %f y %f z %f nx %f ny %f nz %f\n", i, pos[0], pos[1], pos[2], nor[0], nor[1], nor[2]);
    }*/
}

aabox OctreeCreator::getMeshAxisAlignedBoundingBox() {
    return m_aabox;
}

bool OctreeCreator::createSubtree(octree<mesh*>* pNode, mesh m, aabox box, int depth) {
    printf("cre\n");
    pNode[0] = octree<mesh*>();
    pNode->m_node = new mesh(box.cull(m));
    
    if(pNode->m_node->getTriangleCount()) {
        depth--;
        
        if(depth>=0) {
            float3 half_size = box.getSizes()/2.0f;
            float4 centre = box.getCentre();
            float4 corner = box.getCorner();
            
            pNode->allocateChildren();
            
            for(int i = 0; i < 8; i++) {
                float4 new_corner = float4(0.0f, 0.0f, 0.0f, 1.0f);
                
                i & octree<mesh>::X ? new_corner.setX(centre[0]): new_corner.setX(corner[0]);
                i & octree<mesh>::Y ? new_corner.setY(centre[1]): new_corner.setY(corner[1]);
                i & octree<mesh>::Z ? new_corner.setZ(centre[2]): new_corner.setZ(corner[2]);
                
                if( createSubtree(&(pNode->m_children[i]), *(pNode->m_node), aabox(new_corner, half_size), depth)) {
                    pNode->m_children_flag |= (1<<i);
                }
            }
        }
        
        return true;
    }
    return false;
}