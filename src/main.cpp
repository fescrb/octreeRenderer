#include "DeviceManager.h"
#include "DataManager.h"

#include "Octree.h"
#include "Image.h"

using namespace std;

int main() {
	DeviceManager dev;
	dev.printDeviceInfo();
	
	DataManager manager;

	char* buffer = manager.getOctree()->flatten();

	Image image(32,32);
	image.toBMP("test.bmp");
}
