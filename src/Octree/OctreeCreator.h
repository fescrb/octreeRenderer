#ifndef _OCTREE_CREATOR_H
#define _OCTREE_CREATOR_H

#include "AABox.h"
#include <OctreeStruct.h>

class Octree;
class OctreeNode;

class OctreeCreator {
    public:
        explicit                 OctreeCreator(mesh meshToConvert, int depth = 3);

        void                     render();
        void                     convert();

        aabox                    getMeshAxisAlignedBoundingBox();

    private:
        /**
         * @ret true if it contains triangles, false otherwise
         */
        OctreeNode              *createSubtree(octree<mesh*>* pNode, octree<aabox>* bboxes, mesh m, aabox box, int depth);
        
        void                     renderBBoxSubtree(octree<aabox> subtree);
        
        mesh                     m_mesh;
        aabox                    m_aabox;
        int                      m_depth;
        
        Octree                  *m_octree;
        octree<aabox>           *m_bboxes;
};

#endif //_OCTREE_CREATOR_H
