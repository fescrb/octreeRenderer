#include "ConcreteOctree.h"

#include "OctreeStruct.h"

#include "OctreeHeader.h"
#include "OctreeNode.h"
#include "ConcreteOctreeNode.h"
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
    ConcreteOctreeNode *root = new ConcreteOctreeNode(rootAtts);

    // Create topleft node + atts
    Attributes topleftAtts;
    topleftAtts.setColour(255, 255, 255, 255);
    topleftAtts.setNormal(-0.57445626f, 0.57445626f, -0.57445626f);
    ConcreteOctreeNode *topleft = new ConcreteOctreeNode(topleftAtts);

    // Create topright node + atts
    Attributes toprightAtts;
    toprightAtts.setColour(255, 0, 0, 255);
    toprightAtts.setNormal(0.57445626f, 0.57445626f, -0.57445626f);
    ConcreteOctreeNode *topright = new ConcreteOctreeNode(toprightAtts);

    // Create bottomleft node + atts
    Attributes bottomleftAtts;
    bottomleftAtts.setColour(0, 255, 0, 255);
    bottomleftAtts.setNormal(-0.57445626f, -0.57445626f, -0.57445626f);
    ConcreteOctreeNode *bottomleft = new ConcreteOctreeNode(bottomleftAtts);

    // Create bottomright node + atts
    Attributes bottomrightAtts;
    bottomrightAtts.setColour(0, 0, 255, 255);
    bottomrightAtts.setNormal(0.57445626f, -0.57445626f, -0.57445626f);
    ConcreteOctreeNode *bottomright = new ConcreteOctreeNode(bottomrightAtts);

    octree->m_pRootNode = root;
    root->addChild(topleft, ConcreteOctreeNode::Y | ConcreteOctreeNode::Z );
    root->addChild(topright, ConcreteOctreeNode::X | ConcreteOctreeNode::Y | ConcreteOctreeNode::Z);
    root->addChild(bottomleft, ConcreteOctreeNode::Z);
    root->addChild(bottomright, ConcreteOctreeNode::X | ConcreteOctreeNode::Z);

    octree->m_pHeader = new OctreeHeader(octree, octree->getDepth());

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
    return m_pRootNode->getAttributeSize();
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
    
    printf("This octree has %d nodes \n", numOfNodes);

    unsigned int memoryRequired = 0;

    memoryRequired += numOfNodes * 4; // For children flags + attribute pointer.

    memoryRequired += (numOfNodes - 1) * 4; // For children pointers, minus 1 as root node doesn't need one.

    memoryRequired += numOfNodes * m_pHeader->getAttributeSize(); // For the attributes.

    // Debug cout
    //printf("We need %d bytes of memory\n", memoryRequired);

    char* buffer = (char*) malloc (memoryRequired);

    char* end = m_pRootNode->flatten(buffer, m_pHeader->getOctreeDepth());
    
    unsigned int memoryUsed = end - buffer;

    return Bin(buffer, memoryUsed);
}
