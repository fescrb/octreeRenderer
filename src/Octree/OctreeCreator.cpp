#include "OctreeCreator.h"

#include "AABox.h"

#include "Matrix.h"

OctreeCreator::OctreeCreator(mesh meshToConvert)
:   m_mesh(meshToConvert){
    aabox bounding_box(m_mesh);
    
    // We need to centre the mesh at the origin.
    float4 off_centre_difference = float4() - bounding_box.getCentre();
    
    m_mesh = float4x4::translationMatrix(off_centre_difference[0], off_centre_difference[1], off_centre_difference[2]) * m_mesh;
}
