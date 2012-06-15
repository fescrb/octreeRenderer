#include "ConcreteOctreeNode.h"

ConcreteOctreeNode::ConcreteOctreeNode() {
    cleanChildrenPointers();
}

ConcreteOctreeNode::ConcreteOctreeNode(Attributes att) {
    cleanChildrenPointers();
    setAttributes(att);
}

void ConcreteOctreeNode::addChild(OctreeNode* node, unsigned int position_flag) {
    m_vChildren[position_flag] = node;
    numberOfChildren++;
}

OctreeNode* ConcreteOctreeNode::getChildAt(int index) {
    return m_vChildren[index];
}

void ConcreteOctreeNode::cleanChildrenPointers() {
    for(int i = 0; i < 8; i++) {
        m_vChildren[i] = 0;
    }
    numberOfChildren = 0;
}

void ConcreteOctreeNode::setAttributes(Attributes att) {
    m_attributes = att;
}

Attributes ConcreteOctreeNode::getAttributes() {
    return m_attributes;
}

unsigned int ConcreteOctreeNode::getAttributeSize() {
    return getAttributes().getSize();
}

float4 ConcreteOctreeNode::getColour() {
    return getAttributes().getColour();
}

float4 ConcreteOctreeNode::getNormal() {
    return getAttributes().getNormal();
}

unsigned int ConcreteOctreeNode::getDepth() {
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

unsigned int ConcreteOctreeNode::getNumberOfNodes() {
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

char* ConcreteOctreeNode::flatten(char* buffer, int depth) {
    int* buffer_int = (int*) buffer;

    // Create children flag. Then write it.
    char flags = 0;
    int positions = 0;
    int counter = 0;
    for(int i = 0; i < 8; i++) {
        if(m_vChildren[i]){
            flags |= ( 1 << i );
            switch(i) {
                case 1:
                    positions |= counter;
                    break;
                case 2:
                    positions |= (counter << (1));
                    break;
                case 3:
                    positions |= (counter << (3));
                    break;
                case 4:
                    positions |= (counter << (5));
                    break;
                case 5:
                    positions |= (counter << (8));
                    break;
                case 6:
                    positions |= (counter << (11));
                    break;
                case 7:
                    positions |= (counter << (14));
                    break;
                default:
                    break;
            }
            counter ++;
        }
    }
    buffer_int[0] = 0;
    buffer[3] = flags ;
    if(depth > 4) {
        buffer[2] = ((unsigned char)numberOfChildren+1)<<4;
    } else {
        buffer[2] = ((unsigned char)(numberOfChildren/2) + (numberOfChildren%2) + 1)<<4;
        buffer[2] |= 2;
    }
    buffer_int[0] |= positions;

    //printf("flags %d number of children %d positions %d buffer int %d\n", flags, numberOfChildren, positions, buffer_int[0]);

    // Find out where we will write the attributes. Then write
    char* end;
    if(depth > 4)
        end = buffer + ((numberOfChildren + 1 ) *4);
    else
        end = buffer + (((numberOfChildren/2) + (numberOfChildren%2) + 1 ) *4);
    int* attr_location = (int*)end;
    end = m_attributes.flatten(end);

    buffer_int++;
    if(attr_location!=buffer_int)
        buffer_int[0] = 0;
    unsigned short* buffer_short = (unsigned short*) buffer_int;

    if(depth > 4) {
        for(int i = 0; i < 8; i++) {
            if(m_vChildren[i]){
                buffer_int[0] = (int*)end - buffer_int;
                //printf("diff %d\n", buffer_int[0]);
                buffer_int++;
                end = m_vChildren[i]->flatten(end, depth-1);
            }
        }
    } else {
        int cur_pos = 0;
        for(int i = 0; i < 8; i++) {
            if(m_vChildren[i]){
                buffer_short[0] = (int*)end - buffer_int;
                //printf("diff %d at depth %d child %d out of %d int is %d cur_pos %d mod 2 %d\n", buffer_short[0], depth, i, numberOfChildren, buffer_int[0],cur_pos,cur_pos%2);
                if(cur_pos%2) {
                    buffer_int++;
                    if(attr_location!=buffer_int)
                        buffer_int[0] = 0;
                }
                buffer_short++;
                end = m_vChildren[i]->flatten(end, depth-1);
                cur_pos++;
            }
        }
    }

    return end;
}
