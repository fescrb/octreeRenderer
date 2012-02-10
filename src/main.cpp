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

	renderInfo.eyePos.setX(0); //x
	renderInfo.eyePos.setY(0); //y
	renderInfo.eyePos.setZ(-256.0f); //z

	renderInfo.viewDir.setX(0); //x
	renderInfo.viewDir.setY(0); //y
	renderInfo.viewDir.setZ(1.0f); //z

	renderInfo.eyePlaneDist = 1.0f; //Parallel projection, neither of these matter.
	renderInfo.fov = 1.0f;
	
	renderInfo.maxOctreeDepth = manager.getMaxOctreeDepth();
	char* frame = manager.renderFrame(&dev, renderInfo);

	Image image(renderInfo.resolution[0], renderInfo.resolution[1], Image::RGB, frame);
	image.toBMP("test.bmp");
}
