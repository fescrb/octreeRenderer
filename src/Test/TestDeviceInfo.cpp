#include "TestDeviceInfo.h"

#include <fstream>

using namespace std;

TestDeviceInfo::TestDeviceInfo() {
	
	ifstream cpuinfo;

	char line[256];

	cpuinfo.open("/proc/cpuinfo", ifstream::in);

	//TODO
	/*while(!cpuinfo.eof()) {
		cpuinfo.getline(line,256);


	}*/
}

TestDeviceInfo::~TestDeviceInfo() {

}
		
void TestDeviceInfo::printInfo() {

}

char* TestDeviceInfo::getName() {
    return NULL;
}