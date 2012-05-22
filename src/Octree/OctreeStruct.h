#ifndef _OCTREE_SCTRUCT_H
#define _OCTREE_SCTRUCT_H

#include <cstdlib>

template<class t>
struct octree {
    public:
                         octree() : m_children(0),m_node(t()){}
        
        enum PositionFlags {
            X = 1,
            Y = 2,
            Z = 4
        };
        
        void             allocateChildren() {
            m_children = (octree<t>*)malloc((sizeof(octree<t>)*8)+1);
        }
        
        bool             hasChildAt(int position) {
            return m_children_flag & ( 1 << position);
        }
        
        int              m_children_flag;
        octree<t>       *m_children;
        t                m_node;
};

#endif //_OCTREE_SCTRUCT_H