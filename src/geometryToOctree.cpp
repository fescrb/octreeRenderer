#include "AABox.h"

#include "OBJFileReader.h"
#include "OctreeCreator.h"
#include "GeometryOctreeWindow.h"
#include "OctreeWriter.h"

#include <cstdio>
#include <cstring>

void printUsage() {
    printf("Usage:\ngeometryToOctree <inputfile> <outputname> [options]\nArguments in [] are optional.\n");
}

void printHelp() {
    printUsage();
    printf("\n");
    // Print options and such.
}

int main(int argc, char** argv) {
    if(argc < 3) {
        printUsage();
        printf("Type geometryToOctree -help for more info.\n");
        exit(1);
    }
    
    int depth = 1;
    
    if(argc == 4) {
        depth = atoi(argv[3]);
    }

    if(!strcmp(argv[1], "-help")) {
        //Printout help and exit.
        printHelp();
        exit(0);
    }

    OBJFileReader* objFile = new OBJFileReader(argv[1]);

    OctreeCreator* octreeCreator = new OctreeCreator(objFile->getMesh(), depth);
    OctreeWriter* octreeWriter = new OctreeWriter(octreeCreator, argv[2]);

    int2 dim(800, 600);

    GeometryOctreeWindow* window = new GeometryOctreeWindow(argc, argv, dim, octreeCreator, octreeWriter);
    
    window->run();

    return 0;
}
