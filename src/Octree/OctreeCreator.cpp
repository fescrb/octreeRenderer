#include "OctreeCreator.h"

#include "AABox.h"

#include "Matrix.h"

OctreeCreator::OctreeCreator(mesh meshToConvert)
:   m_mesh(meshToConvert),
    m_aabox(meshToConvert){
    // We need to centre the mesh at the origin.
    float4 off_centre_difference = float4() - m_aabox.getCentre();

    float4x4 translation_matrix = float4x4::translationMatrix(off_centre_difference[0], off_centre_difference[1], off_centre_difference[2]);

    m_mesh = translation_matrix * m_mesh;
    m_aabox = translation_matrix * m_aabox;
}

aabox OctreeCreator::getMeshAxisAlignedBoundingBox() {
    return m_aabox;
}
