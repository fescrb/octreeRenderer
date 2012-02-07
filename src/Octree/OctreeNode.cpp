#include "OctreeNode.h"

OctreeNode::OctreeNode() {
	cleanChildrenPointers();
}

OctreeNode::OctreeNode(Attributes att) {
	cleanChildrenPointers();
	setAttributes(att);
}

void OctreeNode::addChild(OctreeNode* node, unsigned int position_flag) {
	m_vChildren[position_flag] = node;
	numberOfChildren++;
}

void OctreeNode::cleanChildrenPointers() {
	for(int i = 0; i < 8; i++) {
		m_vChildren[i] = 0;
	}
	numberOfChildren = 0;
}

void OctreeNode::setAttributes(Attributes att) {
	m_attributes = att;
}

unsigned int OctreeNode::getDepth() {
	unsigned int maxDepth = 0;

	for(int i = 0; i < 8; i++) {
		if(m_vChildren[i]){
			unsigned int thisDepth = m_vChildren[i]->getDepth();
			if(thisDepth > maxDepth)
				maxDepth = thisDepth;
		}
	}

	maxDepth++;
	return maxDepth;
}

unsigned int OctreeNode::getNumberOfNodes() {
	unsigned int numberOfNodes = 0;

	for(int i = 0; i < 8; i++) {
		if(m_vChildren[i]){
			numberOfNodes += m_vChildren[i]->getNumberOfNodes();
		}
	}

	numberOfNodes++;

	return numberOfNodes;
}

char* OctreeNode::flatten(char* buffer) {
	int* buffer_int = (int*) buffer;
	short* buffer_short = (short*) buffer;

	char flags = 0;

	for(int i = 0; i < 8; i++) {
		if(m_vChildren[i]){
			flags |= (1 << i);
		}
	}

	buffer[0] = flags;
	buffer_short[1] = numberOfChildren + 1;

	return buffer;
}
