#ifndef _CONCRETE_OCTREE_NODE_H
#define _CONCRETE_OCTREE_NODE_H

#include "Attributes.h"
#include "OctreeNode.h"

class ConcreteOctreeNode
:   public OctreeNode{
    public:
        explicit                 ConcreteOctreeNode();
        explicit                 ConcreteOctreeNode(Attributes att);

        enum PositionFlags {
            X = 1,
            Y = 2,
            Z = 4
        };

        void                     addChild(OctreeNode* node, unsigned int position_flag);
        OctreeNode              *getChildAt(int index);
        void                     cleanChildrenPointers();

        void                     setAttributes(Attributes att);
        Attributes               getAttributes();
        unsigned int             getAttributeSize();
        
        float4                   getColour();
        float4                   getNormal();

        char                    *flatten(char* buffer, int depth);

        unsigned int             getDepth();
        unsigned int             getNumberOfNodes();

    private:
        OctreeNode              *m_vChildren[8];
        unsigned int             numberOfChildren;

        Attributes               m_attributes;
};

#endif //_CONCRETE_OCTREE_NODE_H