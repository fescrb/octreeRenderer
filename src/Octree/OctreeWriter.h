#ifndef _OCTREE_WRITER_H
#define _OCTREE_WRITER_H

class Octree;

class OctreeWriter {
    public:
        explicit             OctreeWriter(Octree *octree, char* name);
        
        void                 write();
        
    private:
        Octree              *m_pOctree;
        char                *m_sDirectory_name;
};

#endif //_OCTREE_WRITER_H