#include "ProgramState.h"
#include "OctreeRendererWindow.h"

#include "DeviceManager.h"

#include "Octree.h"
#include "Image.h"

#include <cstring>

using namespace std;

int main(int argc, char** argv) {
	
    int2 resolution(1024,640);
	//int2 resolution(1600,720);
	
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
    
    window->run();
}
