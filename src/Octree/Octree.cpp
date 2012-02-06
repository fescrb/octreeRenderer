#include "Octree.h"

#include "OctreeNode.h"

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
