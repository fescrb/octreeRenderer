#include "OctreeCreator.h"

#include "AABox.h"

#include "Matrix.h"

#include <cstdio>

OctreeCreator::OctreeCreator(mesh meshToConvert, int depth)
:   m_mesh(meshToConvert),
    m_aabox(meshToConvert){
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
    aabox new_aabox = m_aabox;
    new_aabox.setSizes(new_aabox.getSizes()*0.5f);
    
    mesh res_mesh = new_aabox.cull(m_mesh);
    
    printf("Mesh size %d\n", res_mesh.getTriangleCount());
    
    for(int i = 0; i < res_mesh.getTriangleCount(); i++) {
        triangle tri = res_mesh.getTriangle(i);
        
        vertex v = tri.getVertex(0); float4 pos = v.getPosition(); float4 nor = v.getNormal();
        printf("Triangle %d vertex 1: x %f y %f z %f nx %f ny %f nz %f\n", i, pos[0], pos[1], pos[2], nor[0], nor[1], nor[2]);
        v = tri.getVertex(1); pos = v.getPosition(); nor = v.getNormal();
        printf("Triangle %d vertex 2: x %f y %f z %f nx %f ny %f nz %f\n", i, pos[0], pos[1], pos[2], nor[0], nor[1], nor[2]);
        v = tri.getVertex(2); pos = v.getPosition(); nor = v.getNormal();
        printf("Triangle %d vertex 3: x %f y %f z %f nx %f ny %f nz %f\n", i, pos[0], pos[1], pos[2], nor[0], nor[1], nor[2]);
    }
}

aabox OctreeCreator::getMeshAxisAlignedBoundingBox() {
    return m_aabox;
}
