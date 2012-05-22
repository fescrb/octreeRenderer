#include "OctreeCreator.h"

#include "AABox.h"

#include "Matrix.h"

#include <cstdio>

OctreeCreator::OctreeCreator(mesh meshToConvert, int depth)
:   m_mesh(meshToConvert),
    m_aabox(meshToConvert){
    // We need to centre the mesh at the origin.
    float4 off_centre_difference = float4() - m_aabox.getCentre();
    
    printf("centre %f %f %f\n", m_aabox.getCentre()[0], m_aabox.getCentre()[1], m_aabox.getCentre()[2]);
    printf("corner %f %f %f\n", m_aabox.getCorner()[0], m_aabox.getCorner()[1], m_aabox.getCorner()[2]);
    printf("sizes %f %f %f\n", m_aabox.getSizes()[0], m_aabox.getSizes()[1], m_aabox.getSizes()[2]);
    printf("off centre diff %f %f %f\n", off_centre_difference[0], off_centre_difference[1], off_centre_difference[2]);

    float4x4 translation_matrix = float4x4::translationMatrix(off_centre_difference[0], off_centre_difference[1], off_centre_difference[2]);

    for(int i = 0; i < m_mesh.getTriangleCount(); i++)
    {
        printf("%d vertex 0 (%f %f %f %f) (%f %f %f %f), vertex 1 (%f %f %f %f) (%f %f %f %f), vertex 2 (%f %f %f %f) (%f %f %f %f)\n", i
            ,m_mesh.getTriangle(i)[0].getPosition()[0]
            ,m_mesh.getTriangle(i)[0].getPosition()[1]
            ,m_mesh.getTriangle(i)[0].getPosition()[2]
            ,m_mesh.getTriangle(i)[0].getPosition()[3]
            ,m_mesh.getTriangle(i)[0].getNormal()[0]
            ,m_mesh.getTriangle(i)[0].getNormal()[1]
            ,m_mesh.getTriangle(i)[0].getNormal()[2]
            ,m_mesh.getTriangle(i)[0].getNormal()[3]
            ,m_mesh.getTriangle(i)[1].getPosition()[0]
            ,m_mesh.getTriangle(i)[1].getPosition()[1]
            ,m_mesh.getTriangle(i)[1].getPosition()[2]
            ,m_mesh.getTriangle(i)[1].getPosition()[3]
            ,m_mesh.getTriangle(i)[1].getNormal()[0]
            ,m_mesh.getTriangle(i)[1].getNormal()[1]
            ,m_mesh.getTriangle(i)[1].getNormal()[2]
            ,m_mesh.getTriangle(i)[1].getNormal()[3]
            ,m_mesh.getTriangle(i)[2].getPosition()[0]
            ,m_mesh.getTriangle(i)[2].getPosition()[1]
            ,m_mesh.getTriangle(i)[2].getPosition()[2]
            ,m_mesh.getTriangle(i)[2].getPosition()[3]
            ,m_mesh.getTriangle(i)[2].getNormal()[0]
            ,m_mesh.getTriangle(i)[2].getNormal()[1]
            ,m_mesh.getTriangle(i)[2].getNormal()[2]
            ,m_mesh.getTriangle(i)[2].getNormal()[3]
        );
    }
    
    m_mesh = translation_matrix * m_mesh;
    
    for(int i = 0; i < m_mesh.getTriangleCount(); i++)
    {
        printf("%d vertex 0 (%f %f %f %f) (%f %f %f %f), vertex 1 (%f %f %f %f) (%f %f %f %f), vertex 2 (%f %f %f %f) (%f %f %f %f)\n", i
            ,m_mesh.getTriangle(i)[0].getPosition()[0]
            ,m_mesh.getTriangle(i)[0].getPosition()[1]
            ,m_mesh.getTriangle(i)[0].getPosition()[2]
            ,m_mesh.getTriangle(i)[0].getPosition()[3]
            ,m_mesh.getTriangle(i)[0].getNormal()[0]
            ,m_mesh.getTriangle(i)[0].getNormal()[1]
            ,m_mesh.getTriangle(i)[0].getNormal()[2]
            ,m_mesh.getTriangle(i)[0].getNormal()[3]
            ,m_mesh.getTriangle(i)[1].getPosition()[0]
            ,m_mesh.getTriangle(i)[1].getPosition()[1]
            ,m_mesh.getTriangle(i)[1].getPosition()[2]
            ,m_mesh.getTriangle(i)[1].getPosition()[3]
            ,m_mesh.getTriangle(i)[1].getNormal()[0]
            ,m_mesh.getTriangle(i)[1].getNormal()[1]
            ,m_mesh.getTriangle(i)[1].getNormal()[2]
            ,m_mesh.getTriangle(i)[1].getNormal()[3]
            ,m_mesh.getTriangle(i)[2].getPosition()[0]
            ,m_mesh.getTriangle(i)[2].getPosition()[1]
            ,m_mesh.getTriangle(i)[2].getPosition()[2]
            ,m_mesh.getTriangle(i)[2].getPosition()[3]
            ,m_mesh.getTriangle(i)[2].getNormal()[0]
            ,m_mesh.getTriangle(i)[2].getNormal()[1]
            ,m_mesh.getTriangle(i)[2].getNormal()[2]
            ,m_mesh.getTriangle(i)[2].getNormal()[3]
        );
    }
    m_aabox = translation_matrix * m_aabox;
}

void OctreeCreator::render() {
    m_mesh.render();
}

void OctreeCreator::convert() {
    aabox new_aabox = m_aabox;
    new_aabox.setSizes(new_aabox.getSizes()*0.5f);
    
    printf("Start mesh size %d\n", m_mesh.getTriangleCount());
    printf("BBox (%f %f %f) (%f %f %f)\n",
        new_aabox.getCorner()[0],
        new_aabox.getCorner()[1],
        new_aabox.getCorner()[2],
        new_aabox.getSizes()[0],
        new_aabox.getSizes()[1],
        new_aabox.getSizes()[2]
    );
    
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
