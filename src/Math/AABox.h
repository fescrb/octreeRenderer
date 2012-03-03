#ifndef _A_A_BOX_H
#define _A_A_BOX_H

#include "Mesh.h"

struct aabox {
    public:
        explicit        aabox(const float4& corner, 
                              const float3& sizes) 
                        :   m_corner(corner), 
                            m_sizes(sizes){}
        
        explicit        aabox(const float4& corner1, 
                              const float4& corner2) 
                        :   m_corner(corner1), 
                            m_sizes(corner2[0]-corner1[0], corner2[1]-corner1[1], corner2[2]-corner1[2]){}
                            
        explicit        aabox(const mesh& meshToBound);
                            
                        aabox(const aabox& other) 
                        :   m_corner(other.m_corner), 
                            m_sizes(other.m_sizes){}
        
        // TODO
        mesh            cull(mesh) {
            mesh resultantMesh;
            
            
            return resultantMesh;
        }
        
        inline float4   getCentre() const {
            return m_corner + direction(m_sizes/2.0f);
        }
        
        inline float4   getCorner() const {
            return m_corner;
        }
        
    private:
        float4          m_corner;
        float3          m_sizes;
};

#endif //_A_A_BOX_H