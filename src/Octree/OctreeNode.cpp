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

OctreeNode* OctreeNode::getChildAt(int index) {
    return m_vChildren[index];
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

Attributes OctreeNode::getAttributes() {
    return m_attributes;
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

#include <cstdio>

char* OctreeNode::flatten(char* buffer) {
	int* buffer_int = (int*) buffer;
	short* buffer_short = (short*) buffer;

	// Create children flag. Then write it.
    char flags = 0;
    int positions = 0;
    int counter = 0;
    for(int i = 0; i < 8; i++) {
        if(m_vChildren[i]){
            flags |= ( 1 << i );
            positions |= (counter << (i*3));
            counter ++;
        }
    }
    buffer_int[0] = positions;
    buffer_int++;
    buffer+=sizeof(int);
    
    // Write the number of children. This is the attribute pointer (for now).
    buffer_int[0] = numberOfChildren + 1;
    buffer[3] = flags ;

	// Find out where we will write the attributes. Then write
	char* end = buffer + ((numberOfChildren + 1 ) *4);
	end = m_attributes.flatten(end);

	buffer_int++;

	for(int i = 0; i < 8; i++) {
		if(m_vChildren[i]){
			buffer_int[0] = (int*)end - buffer_int;
			//printf("diff %d\n", buffer_int[0]);
			buffer_int++;
			end = m_vChildren[i]->flatten(end);
		}
	}

	return end;
}
