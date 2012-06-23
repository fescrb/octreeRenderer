#include "ProgramState.h"
#include "OctreeRendererWindow.h"

#include "DeviceManager.h"

#include "Octree.h"
#include "Image.h"

#include <cstring>

#include "PNGReader.h"

using namespace std;

int main(int argc, char** argv) {
	
    //int2 resolution(600,400);
    //int2 resolution(800,600);
    int2 resolution(1024,640);
    //int2 resolution(1280,720);
    //int2 resolution(1368,768);
    //int2 resolution(1400,900);
	//int2 resolution(1600,900);
    
    //PNGReader reader("test.png");
    //reader.readImage();
    //exit(0);
	
	for(int i = 2; i < argc; i++) {
		if(!strcmp(argv[i],"resolution")){
			i++;
			resolution.setX(atoi(argv[i]));
			i++;
			resolution.setY(atoi(argv[i]));
		}
	}
    
    ProgramState *state = new ProgramState(argc, argv, resolution);
    OctreeRendererWindow *window = new OctreeRendererWindow(argc, argv, resolution, state);
	
	state->getDeviceManager()->initialise();
    
    exit(0);
    
    window->run();
}
