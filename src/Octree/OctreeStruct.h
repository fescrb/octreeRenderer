#ifndef _OCTREE_SCTRUCT_H
#define _OCTREE_SCTRUCT_H

#include <cstdlib>

template<class t>
struct octree {
    public:
                         octree() : m_children_flag(0), m_children(0),m_node(t()){}
        
        enum PositionFlags {
            X = 1,
            Y = 2,
            Z = 4
        };
        
        void             allocateChildren() {
            m_children = (octree<t>*)malloc((sizeof(octree<t>)*8)+1);
            for(int i = 0; i < 8; i++) {
                m_children[i] = octree<t>();
            }
        }
        
        bool             hasChildren() {
            return m_children_flag;
        }
        
        bool             hasChildAt(int position) {
            return m_children_flag & ( 1 << position);
        }
        
        void             addChildToFlagAt(int position) {
            m_children_flag |= ( 1 << position);
        }
        
        void             addChildAt(octree<t> child, int position) {
            if(!m_children) 
                allocateChildren();
            m_children[position]=child;
            addChildToFlagAt(position);
        }
        
        int              m_children_flag;
        octree<t>       *m_children;
        t                m_node;
};

#endif //_OCTREE_SCTRUCT_H