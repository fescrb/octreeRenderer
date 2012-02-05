#include "DeviceManager.h"

#include "Image.h"

using namespace std;

int main() {
	DeviceManager dev;
	dev.printDeviceInfo();
	
	Image image(32,32);
	image.toBMP("test.bmp");
}
