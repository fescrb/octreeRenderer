#include "DeviceManager.h"
#include "DataManager.h"
#include "RenderInfo.h"

#include "Octree.h"
#include "Image.h"

using namespace std;

int main() {
	DeviceManager dev;
	dev.printDeviceInfo();
	
	DataManager manager;

	RenderInfo renderInfo;
	renderInfo.resolution[0] = 32;
	renderInfo.resolution[1] = 32;

	renderInfo.eyePos[0] = 0; //x
	renderInfo.eyePos[1] = 0; //y
	renderInfo.eyePos[2] = -256.0f; //z

	renderInfo.viewDir[0] = 0; //x
	renderInfo.viewDir[1] = 0; //y
	renderInfo.viewDir[2] = 1.0f; //z

	renderInfo.eyePlaneDist = 1.0f; //Parallel projection, neither of these matter.
	renderInfo.fov = 1.0f;
	char* frame = manager.renderFrame(&dev, renderInfo);

	Image image(renderInfo.resolution[0], renderInfo.resolution[1], Image::RGB, frame);
	image.toBMP("test.bmp");
}
