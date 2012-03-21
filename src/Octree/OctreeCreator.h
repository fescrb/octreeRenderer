#ifndef _OCTREE_CREATOR_H
#define _OCTREE_CREATOR_H

#include "AABox.h"

class Octree;

class OctreeCreator {
    public:
        explicit                 OctreeCreator(mesh meshToConvert, int depth = 3);

        void                     render();
        void                     convert();

        aabox                    getMeshAxisAlignedBoundingBox();

    private:
        mesh                     m_mesh;
        aabox                    m_aabox;
        
        Octree                  *m_octree;
};

#endif //_OCTREE_CREATOR_H
