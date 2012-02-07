#include "Octree.h"

#include "OctreeNode.h"

#include <cstdlib>
#include <cstdio>

Octree::Octree() {

}

Octree* Octree::getSimpleOctree() {
	Octree *octree = new Octree();
	
	// Create root node + atts
	Attributes rootAtts;
	rootAtts.setAttributes(128, 128, 128, 128);
	OctreeNode *root = new OctreeNode(rootAtts);
	
	// Create topleft node + atts
	Attributes topleftAtts;
	topleftAtts.setAttributes(255, 255, 255, 255);
	OctreeNode *topleft = new OctreeNode(topleftAtts);
	
	// Create topright node + atts
	Attributes toprightAtts;
	toprightAtts.setAttributes(255, 0, 0, 255);
	OctreeNode *topright = new OctreeNode(toprightAtts);
	
	// Create bottomleft node + atts
	Attributes bottomleftAtts;
	bottomleftAtts.setAttributes(0, 255, 0, 255);
	OctreeNode *bottomleft = new OctreeNode(bottomleftAtts);
	
	// Create bottomright node + atts
	Attributes bottomrightAtts;
	bottomrightAtts.setAttributes(0, 0, 0, 255);
	OctreeNode *bottomright = new OctreeNode(bottomrightAtts);
	
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

char* Octree::flatten() {
	unsigned int numOfNodes = getNumberOfNodes();

	unsigned int memoryRequired = 0;

	memoryRequired += numOfNodes * 4; // For children flags + attribute pointer.

	memoryRequired += (numOfNodes - 1) * 4; // For children pointers, minus 1 as root node doesn't need one.

	memoryRequired += numOfNodes * 4; // For the attributes.

	printf("We need %d bytes of memory\n", memoryRequired);

	char* buffer = (char*) malloc (memoryRequired);

	m_pRootNode->flatten(buffer);

	return buffer;
}
