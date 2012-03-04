#ifndef _OCTREE_CREATOR_H
#define _OCTREE_CREATOR_H

#include "AABox.h"

class OctreeCreator {
    public:
        explicit                 OctreeCreator(mesh meshToConvert);

        void                     render();
        void                     convert();

        aabox                    getMeshAxisAlignedBoundingBox();

    private:
        mesh                     m_mesh;
        aabox                    m_aabox;
};

#endif //_OCTREE_CREATOR_H
