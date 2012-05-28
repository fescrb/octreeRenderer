#include "ConcreteOctree.h"

#include "OctreeStruct.h"

#include "OctreeHeader.h"
#include "OctreeNode.h"
#include "OctreeSegment.h"

#include <cstdlib>
#include <cstdio>

ConcreteOctree::ConcreteOctree() {

}

Octree* ConcreteOctree::getSimpleOctree() {
    ConcreteOctree *octree = new ConcreteOctree();
    
    // Create root node + atts
    Attributes rootAtts;
    rootAtts.setColour(128, 128, 128, 128);
    rootAtts.setNormal(0.0f, 0.0f, -1.0f);
    OctreeNode *root = new OctreeNode(rootAtts);
    
    // Create topleft node + atts
    Attributes topleftAtts;
    topleftAtts.setColour(255, 255, 255, 255);
    topleftAtts.setNormal(-0.57445626f, 0.57445626f, -0.57445626f);
    OctreeNode *topleft = new OctreeNode(topleftAtts);
    
    // Create topright node + atts
    Attributes toprightAtts;
    toprightAtts.setColour(255, 0, 0, 255);
    toprightAtts.setNormal(0.57445626f, 0.57445626f, -0.57445626f);
    OctreeNode *topright = new OctreeNode(toprightAtts);
    
    // Create bottomleft node + atts
    Attributes bottomleftAtts;
    bottomleftAtts.setColour(0, 255, 0, 255);
    bottomleftAtts.setNormal(-0.57445626f, -0.57445626f, -0.57445626f);
    OctreeNode *bottomleft = new OctreeNode(bottomleftAtts);
    
    // Create bottomright node + atts
    Attributes bottomrightAtts;
    bottomrightAtts.setColour(0, 0, 255, 255);
    bottomrightAtts.setNormal(0.57445626f, -0.57445626f, -0.57445626f);
    OctreeNode *bottomright = new OctreeNode(bottomrightAtts);
    
    octree->m_pRootNode = root;
    root->addChild(topleft, OctreeNode::Y | OctreeNode::Z );
    root->addChild(topright, OctreeNode::X | OctreeNode::Y | OctreeNode::Z);
    root->addChild(bottomleft, OctreeNode::Z);
    root->addChild(bottomright, OctreeNode::X | OctreeNode::Z);
    
    octree->m_pHeader = new OctreeHeader(octree);
    
    renderinfo info;
    
    info.eyePos.setX(0); //x
    info.eyePos.setY(0); //y
    info.eyePos.setZ(-256.0f); //z
    
    info.viewDir.setX(0); //x
    info.viewDir.setY(0); //y
    info.viewDir.setZ(1.0f); //z
    
    info.up.setX(0); //x
    info.up.setY(1.0f); //y
    info.up.setZ(0); //z
    
    info.eyePlaneDist = 1.0f; //Parallel projection, neither of these matter.
    info.fov = 1.0f;
    
    info.lightPos = float3(128.0f,128.0f,-128.0f);
    info.lightBrightness = 256;
    
    octree->setInitialRenderInfo(info);
    
    return octree;
}

unsigned int ConcreteOctree::getDepth() {
    return m_pRootNode->getDepth();
}

unsigned int ConcreteOctree::getAttributeSize() {
    return m_pRootNode->getAttributes().getSize();
}

unsigned int ConcreteOctree::getNumberOfNodes() {
    return m_pRootNode->getNumberOfNodes();
}



void ConcreteOctree::setInitialRenderInfo(renderinfo info) {
    m_initial_renderinfo = info;
}

Bin ConcreteOctree::getHeader() {
    return m_pHeader->flatten();
}

Bin ConcreteOctree::getRoot() {
    return flatten();
}

Bin ConcreteOctree::flatten() {
    unsigned int numOfNodes = getNumberOfNodes();

    unsigned int memoryRequired = 0;

    memoryRequired += numOfNodes * 8; // For children flags + attribute pointer.

    memoryRequired += (numOfNodes - 1) * 4; // For children pointers, minus 1 as root node doesn't need one.

    memoryRequired += numOfNodes * m_pHeader->getAttributeSize(); // For the attributes.

    // Debug cout
    //printf("We need %d bytes of memory\n", memoryRequired);

    char* buffer = (char*) malloc (memoryRequired);

    m_pRootNode->flatten(buffer);

    return Bin(buffer, memoryRequired);
}
