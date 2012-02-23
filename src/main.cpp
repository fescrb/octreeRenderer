#include "ProgramState.h"
#include "Window.h"

#include "DeviceManager.h"

#include "Octree.h"
#include "Image.h"

#include <cstring>

using namespace std;

int main(int argc, char** argv) {
	
    int2 resolution(800,400);
	
	for(int i = 1; i < argc; i++) {
		if(!strcmp(argv[i],"resolution")){
			i++;
			resolution.setX(atoi(argv[i]));
			i++;
			resolution.setY(atoi(argv[i]));
		}
	}
    
    ProgramState *state = new ProgramState(argc, argv);
    Window *window = new Window(argc, argv, resolution, state);
	
	state->getDeviceManager()->detectDevices();
    
    window->run();

	//char* frame = manager.renderFrame(&dev, renderInfo);
}
