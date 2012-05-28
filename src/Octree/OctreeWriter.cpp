#include "OctreeWriter.h"

#include "Octree.h"

#include "Path.h"
#include "BinWriter.h"
#include "RenderInfoWriter.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

OctreeWriter::OctreeWriter(Octree *octree, char* name)
:   m_pOctree(octree) {
    m_sDirectory_name = (char*)malloc(sizeof(char)*(strlen(name)+5)+1);
    sprintf(m_sDirectory_name, "%s.voct",name);
}

//! TODO
void OctreeWriter::write() {
    if(!make_directory(m_sDirectory_name)) {
        BinWriter header_writer = BinWriter(m_pOctree->getHeader(), m_sDirectory_name, "header");
        header_writer.writeAll();
        BinWriter root_writer = BinWriter(m_pOctree->getRoot(), m_sDirectory_name, "0");
        root_writer.writeAll();
        RenderInfoWriter info_writer = RenderInfoWriter(m_pOctree->getInitialRenderInfo(), m_sDirectory_name);
        info_writer.writeAll();
    }
}