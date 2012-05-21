#include "Octree.h"

#include "OctreeHeader.h"
#include "OctreeNode.h"
#include "OctreeSegment.h"

#include <cstdlib>
#include <cstdio>

Octree::Octree() {

}

Octree* Octree::getSimpleOctree() {
	Octree *octree = new Octree();
    octree->m_pHeader = new OctreeHeader();
	
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
	
    octree->m_pHeader->setAttributeSize(rootAtts.getSize());
    
	octree->m_pRootNode = root;
	root->addChild(topleft, OctreeNode::Y | OctreeNode::Z );
	root->addChild(topright, OctreeNode::X | OctreeNode::Y | OctreeNode::Z);
	root->addChild(bottomleft, OctreeNode::Z);
	root->addChild(bottomright, OctreeNode::X | OctreeNode::Z);
	
	return octree;
}

unsigned int Octree::getDepth() {
	return m_pRootNode->getDepth();
}

unsigned int Octree::getNumberOfNodes() {
	return m_pRootNode->getNumberOfNodes();
}

Bin Octree::getHeader() {
    return m_pHeader->flatten();
}

Bin Octree::flatten() {
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
