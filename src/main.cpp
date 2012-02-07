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
	char* frame = manager.renderFrame(&dev, renderInfo);

	Image image(renderInfo.resolution[0], renderInfo.resolution[1], Image::RGB, frame);
	image.toBMP("test.bmp");
}
