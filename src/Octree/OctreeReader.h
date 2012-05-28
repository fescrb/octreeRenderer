#ifndef _OCTREE_READER_H
#define _OCTREE_READER_H

#include "Octree.h"

class OctreeReader
:   public Octree {
    public:
        explicit                 OctreeReader(char* name);
        
        Bin                      getHeader();
        Bin                      getRoot();
        
    private:
        char                    *m_sPath;
};

#endif //_OCTREE_READER_H