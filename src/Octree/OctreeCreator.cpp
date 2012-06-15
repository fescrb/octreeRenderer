#include "OctreeCreator.h"

#include "Octree.h"
#include "OctreeNode.h"
#include "ConcreteOctreeNode.h"
#include "OctreeHeader.h"

#include "AABox.h"
#include "Matrix.h"

#include "MathUtil.h"

#include <cstdio>

OctreeCreator::OctreeCreator(mesh meshToConvert, int depth, bool keep_aaboxes)
:   m_converted(false),
    m_mesh(meshToConvert),
    m_aabox(meshToConvert),
    m_depth(depth),
    m_bboxes(0),
    m_renderVoxels(false),
    m_keep_aaboxes(keep_aaboxes){

    // We need to centre the mesh at the origin.
    float4 off_centre_difference = float4(0.0f,0.0f,0.0f,2.0f) - m_aabox.getCentre();

    float4x4 translation_matrix = float4x4::translationMatrix(off_centre_difference[0], off_centre_difference[1], off_centre_difference[2]);

    /*printf("off_centre %f %f %f %f\nsize %d\n", off_centre_difference[0], off_centre_difference[1], off_centre_difference[2], off_centre_difference[3], m_mesh.getTriangleCount());
    printf("before aabox corner %f %f %f %f size %f %f %f\n", m_aabox.getCorner()[0], m_aabox.getCorner()[1], m_aabox.getCorner()[2], m_aabox.getCorner()[3],
           m_aabox.getSizes()[0],m_aabox.getSizes()[1],m_aabox.getSizes()[2]);

    /*for(int i = 0; i < m_mesh.getTriangleCount(); i++) {
        for(int j = 0; j < 3; j++) {
            float4 pos = m_mesh.getTriangle(i).getVertex(j).getPosition();
            float4 nor = m_mesh.getTriangle(i).getVertex(j).getNormal();
            printf("pos (%f %f %f %f) nor (%f %f %f %f)\n", pos[0], pos[1], pos[2], pos[3], nor[0], nor[1], nor[2], nor[3]);
        }
    }*/

    m_mesh = translation_matrix * m_mesh;
    m_aabox = translation_matrix * m_aabox;

    /*printf(" after aabox corner %f %f %f %f size %f %f %f\n", m_aabox.getCorner()[0], m_aabox.getCorner()[1], m_aabox.getCorner()[2], m_aabox.getCorner()[3],
           m_aabox.getSizes()[0],m_aabox.getSizes()[1],m_aabox.getSizes()[2]);

    /*for(int i = 0; i < m_mesh.getTriangleCount(); i++) {
        for(int j = 0; j < 3; j++) {
            float4 pos = m_mesh.getTriangle(i).getVertex(j).getPosition();
            float4 nor = m_mesh.getTriangle(i).getVertex(j).getNormal();
            printf("pos (%f %f %f %f) nor (%f %f %f %f)\n", pos[0], pos[1], pos[2], pos[3], nor[0], nor[1], nor[2], nor[3]);
        }
    }*/

    float3 sizes = float3(m_aabox.getLongestSize());
    m_aabox = aabox(m_aabox.getCentre() - (direction(sizes)/2.0f), sizes);

    renderinfo initial;

    float3 total_octree_volume = float3(OCTREE_ROOT_HALF_SIZE*2.0f,OCTREE_ROOT_HALF_SIZE*2.0f,OCTREE_ROOT_HALF_SIZE*2.0f);
    float center_distance_to_camera = mag(total_octree_volume);

    initial.fov = 30.0f;
    initial.up = normalize(float3(-1.0f,1.0f,-1.0f));
    initial.eyePos = float3(1.0f * center_distance_to_camera, 1.0f * center_distance_to_camera, 1.0f * center_distance_to_camera);
    initial.viewPortStart = float3();
    initial.viewStep = float3();
    initial.viewDir = normalize(float3() - initial.eyePos);
    initial.eyePlaneDist = center_distance_to_camera/2.0f;

    initial.lightPos = float3(OCTREE_ROOT_HALF_SIZE*2.0f, 0.0f, 0.0f);
    initial.lightBrightness = OCTREE_ROOT_HALF_SIZE;

    setInitialRenderInfo(initial);
}

void OctreeCreator::render() {
    if(!m_renderVoxels)
        m_mesh.render();
    if(m_bboxes) {
        renderBBoxSubtree(*m_bboxes, m_pRootNode);
    }
}

void OctreeCreator::renderBBoxSubtree(octree<aabox> subtree, OctreeNode *node) {
    if(!subtree.hasChildren()) {
        if( m_renderVoxels ) {
            subtree.m_node.renderVoxel(node->getColour(), node->getNormal());
        } else
            subtree.m_node.render();
    } else {
        for (int i = 0; i < 8; i++) {
            if(subtree.hasChildAt(i))
                renderBBoxSubtree(subtree.m_children[i], node->getChildAt(i));
        }
    }
}

void OctreeCreator::toggleRenderMode() {
    m_renderVoxels = !m_renderVoxels;
}

bool OctreeCreator::isConverted() {
    return m_converted;
}

void OctreeCreator::convert() {
    if(m_keep_aaboxes) {
        m_bboxes = (octree<aabox>*) malloc(sizeof(octree<aabox>));
        m_bboxes[0] = octree<aabox>();
    }
    float3 half_size = m_aabox.getSizes()/2.0f;
    float4 centre = m_aabox.getCentre();
    float4 corner = m_aabox.getCorner();


    OctreeNode *root = createSubtree(m_bboxes, m_mesh, m_aabox, m_depth);

    m_pRootNode = root;

    this->m_pHeader = new OctreeHeader(this, m_depth);

    m_converted = true;


    /*printf("Root node col %f %f %f %f normal %f %f %f\n",
        root->getAttributes().getColour()[0],
        root->getAttributes().getColour()[1],
        root->getAttributes().getColour()[2],
        root->getAttributes().getColour()[3],
        root->getAttributes().getNormal()[0],
        root->getAttributes().getNormal()[1],
        root->getAttributes().getNormal()[2]
    );*/

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

OctreeNode* OctreeCreator::createSubtree(octree<aabox>* bboxes, mesh m, aabox box, int depth) {
    mesh* this_m = new mesh(box.cull(m));
    if(bboxes)
        bboxes->m_node = box;

    if(this_m->getTriangleCount()) {
        depth--;

        ConcreteOctreeNode *this_node = new ConcreteOctreeNode();
        bool has_children = false;
        int children = 0;

        if(depth>=0) {
            float3 half_size = box.getSizes()/2.0f;
            float4 centre = box.getCentre();
            float4 corner = box.getCorner();

            if(bboxes)
                bboxes->allocateChildren();

            float4 colour = float4();
            float4 normal = float4();

            #pragma omp parallel for
            for(int i = 0; i < 8; i++) {
                float4 new_corner = float4(0.0f, 0.0f, 0.0f, 1.0f);

                i & octree<mesh>::X ? new_corner.setX(centre[0]): new_corner.setX(corner[0]);
                i & octree<mesh>::Y ? new_corner.setY(centre[1]): new_corner.setY(corner[1]);
                i & octree<mesh>::Z ? new_corner.setZ(centre[2]): new_corner.setZ(corner[2]);

                octree<aabox>* child_box = 0;
                if(bboxes)
                    child_box = &(bboxes->m_children[i]);
                OctreeNode* child_node = createSubtree(child_box, *(this_m), aabox(new_corner, half_size), depth);

                if(child_node) {
                    if(bboxes)
                        bboxes->addChildToFlagAt(i);
                    has_children = true;
                    this_node->addChild(child_node, i);
                    children++;
                }
                //printf("----------------\n");
            }

            if(has_children) {

                for(int i = 0; i < 8; i++) {

                    OctreeNode *child_node = this_node->getChildAt(i);

                    if(child_node) {
                        has_children = true;
                        colour+=child_node->getColour();
                        /*printf("child_colour %f %f %f %f\n",
                               child_node->getAttributes().getColour()[0],
                               child_node->getAttributes().getColour()[1],
                               child_node->getAttributes().getColour()[2],
                               child_node->getAttributes().getColour()[3]
                        );*/
                        normal+=child_node->getNormal();
                        /*printf("child normal %f %f %f\n",
                               child_node->getAttributes().getNormal()[0],
                               child_node->getAttributes().getNormal()[1],
                               child_node->getAttributes().getNormal()[2]
                        );*/
                    }
                    //printf("----------------\n");
                }

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
            double total_area = 0;
            for(int i = 0; i < this_m->getTriangleCount(); i++) {
                double area = this_m->getTriangle(i).getSurfaceArea();
                total_area+=area;
                colour += this_m->getTriangle(i).getAverageColour() * area;
                normal += this_m->getTriangle(i).getAverageNormal() * area;
            }

            colour/=total_area;
            normal/=total_area;
            normal=normalize(normal);

            //printf("this col %f %f %f %f nor %f %f %f\n-----------\n",colour[0],colour[1],colour[2],colour[3],normal[0],normal[1],normal[2]);

            Attributes atts;
            atts.setColour(colour);
            atts.setNormal(normal[0],normal[1], normal[2]);

            this_node->setAttributes(atts);
        }
        delete this_m;
        return this_node;
    }
    return NULL;
}
