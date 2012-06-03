#ifndef _OCTREE_CREATOR_H
#define _OCTREE_CREATOR_H

#include "AABox.h"
#include <OctreeStruct.h>

#include "ConcreteOctree.h"

class OctreeNode;

class OctreeCreator 
:   public ConcreteOctree {
    public:
        explicit                 OctreeCreator(mesh meshToConvert, int depth, bool keep_aaboxes);

        void                     render();
        void                     convert();
        
        bool                     isConverted();

        aabox                    getMeshAxisAlignedBoundingBox();
        
        void                     toggleRenderMode();

    private:
        /**
         * @ret true if it contains triangles, false otherwise
         */
        OctreeNode              *createSubtree(octree<aabox>* bboxes, mesh m, aabox box, int depth);
        
        void                     renderBBoxSubtree(octree<aabox> subtree, OctreeNode *node);
        
        bool                     m_converted;
        
        bool                     m_renderVoxels;
        
        mesh                     m_mesh;
        aabox                    m_aabox;
        int                      m_depth;
        
        bool                     m_keep_aaboxes;
        octree<aabox>           *m_bboxes;
};

#endif //_OCTREE_CREATOR_H
