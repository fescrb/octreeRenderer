#include "OctreeCreator.h"

#include "Octree.h"
#include "OctreeNode.h"

#include "AABox.h"
#include "Matrix.h"

#include "MathUtil.h"

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
    
    
    OctreeNode *root = createSubtree(&mesh_octree, m_mesh, m_aabox, m_depth);
   
    printf("Root node col %f %f %f %f normal %f %f %f\n",
        root->getAttributes().getColour()[0],
        root->getAttributes().getColour()[1],
        root->getAttributes().getColour()[2],
        root->getAttributes().getColour()[3],
        root->getAttributes().getNormal()[0],
        root->getAttributes().getNormal()[1],
        root->getAttributes().getNormal()[2]
    );
    
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

OctreeNode* OctreeCreator::createSubtree(octree<mesh*>* pNode, mesh m, aabox box, int depth) {
    printf("cre\n");
    pNode[0] = octree<mesh*>();
    pNode->m_node = new mesh(box.cull(m));
    
    if(pNode->m_node->getTriangleCount()) {
        depth--;
        
        OctreeNode *this_node = new OctreeNode();
        bool has_children = false;
        int children = 0;
        
        if(depth>=0) {
            float3 half_size = box.getSizes()/2.0f;
            float4 centre = box.getCentre();
            float4 corner = box.getCorner();
            
            pNode->allocateChildren();
            
            float4 colour = float4();
            float4 normal = float4();
            
            for(int i = 0; i < 8; i++) {
                float4 new_corner = float4(0.0f, 0.0f, 0.0f, 1.0f);
                
                i & octree<mesh>::X ? new_corner.setX(centre[0]): new_corner.setX(corner[0]);
                i & octree<mesh>::Y ? new_corner.setY(centre[1]): new_corner.setY(corner[1]);
                i & octree<mesh>::Z ? new_corner.setZ(centre[2]): new_corner.setZ(corner[2]);
                
                OctreeNode* child_node = createSubtree(&(pNode->m_children[i]), *(pNode->m_node), aabox(new_corner, half_size), depth);
                
                if(child_node) {
                    pNode->m_children_flag |= (1<<i);
                    has_children = true;
                    colour+=child_node->getAttributes().getColour();
                    printf("child_colour %f %f %f %f\n",
                           child_node->getAttributes().getColour()[0],
                           child_node->getAttributes().getColour()[1],
                           child_node->getAttributes().getColour()[2],
                           child_node->getAttributes().getColour()[3]
                    );
                    normal+=child_node->getAttributes().getNormal();
                    printf("child normal %f %f %f\n",
                           child_node->getAttributes().getNormal()[0],
                           child_node->getAttributes().getNormal()[1],
                           child_node->getAttributes().getNormal()[2]
                    );
                    this_node->addChild(child_node, i);
                    children++;
                }
            }
            
            if(has_children) {
                normal/=(float)children;
                colour/=(float)children;
                
                Attributes atts;
                atts.setColour(colour);
                atts.setNormal(normal[0],normal[1], normal[2]);
                
                this_node->setAttributes(atts);
            }
        } 
        
        if(!has_children){
            float4 colour = float4();
            float4 normal = float4();
            float total_area = 0;
            for(int i = 0; i < pNode->m_node->getTriangleCount(); i++) {
                float area = pNode->m_node->getTriangle(i).getSurfaceArea();
                total_area+=area;
                colour += pNode->m_node->getTriangle(i).getAverageColour() * area;
                normal += pNode->m_node->getTriangle(i).getAverageNormal() * area;
            }
            colour/=total_area;
            normal/=total_area;
            normal=normalize(normal);
            
            printf("child node col %f %f %f %f nor %f %f %f\n-----------\n",colour[0],colour[1],colour[2],colour[3],normal[0],normal[1],normal[2]);
            
            Attributes atts;
            atts.setColour(colour);
            atts.setNormal(normal[0],normal[1], normal[2]);
            
            this_node->setAttributes(atts);
        }
        
        return this_node;
    }
    return NULL;
}