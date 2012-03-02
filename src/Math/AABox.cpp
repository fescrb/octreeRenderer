#include "AABox.h"

aabox::aabox(const mesh& meshToBound) {
    std::vector<vertex> max_min = meshToBound.getOuterMostVertices();
    float x_max = max_min[0].getPosition().getX();
    float x_min = max_min[1].getPosition().getX();
    float y_max = max_min[2].getPosition().getY();
    float y_min = max_min[3].getPosition().getY();
    float z_max = max_min[4].getPosition().getZ();
    float z_min = max_min[5].getPosition().getZ();
    
    m_corner = position(float3(x_min, y_min, y_max));
    m_sizes = float3(x_max-x_min, y_max-y_min, z_max-z_min);
}