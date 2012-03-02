#ifndef _OCTREE_CREATOR_H
#define _OCTREE_CREATOR_H

#include "Mesh.h"

class OctreeCreator {
    public:
        explicit                 OctreeCreator(mesh meshToConvert);
        
        void                     render();
        void                     convert();
        
    private:
        mesh                     m_mesh;
};

#endif //_OCTREE_CREATOR_H