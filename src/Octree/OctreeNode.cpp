#include "OctreeNode.h"

OctreeNode::OctreeNode() {
	
}

OctreeNode::OctreeNode(Attributes att) {
	setAttributes(att);
}

void OctreeNode::addChild(OctreeNode* node, unsigned int position_flag) {
	m_vChildren[position_flag] = node;
}

void OctreeNode::setAttributes(Attributes att) {
	m_attributes = att;
}
