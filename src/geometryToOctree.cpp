#include "AABox.h"

#include "OBJFileReader.h"

#include <cstdio>
#include <cstring>

void printUsage() {
    printf("Usage:\ngeometryToOctree <inputfile> [outputname] [options]\nArguments in [] are optional.\n");
}

void printHelp() {
    printUsage();
    printf("\n");
    // Print options and such.
}

int main(int argc, char** argv) {
    if(argc < 2) {
        printUsage();
        printf("Type geometryToOctree -help for more info.\n");
        exit(1);
    }
    
    if(!strcmp(argv[1], "-help")) {
        //Printout help and exit.
        printHelp();
        exit(0);
    }
    
    OBJFileReader* objFile = new OBJFileReader(argv[1]);
    
    objFile->getMesh();
    
    return 0;
}