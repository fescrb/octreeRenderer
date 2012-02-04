#include "OctreeNode.h"

OctreeNode::OctreeNode() {

}

void OctreeNode::addChild(OctreeNode* node, unsigned int position_flag) {
	m_vChildren[position_flag] = node;
}
