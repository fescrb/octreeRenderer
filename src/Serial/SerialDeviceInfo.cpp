#include "SerialDeviceInfo.h"

#include <fstream>

using namespace std;

SerialDeviceInfo::SerialDeviceInfo() {
	
	ifstream cpuinfo;

	char line[256];

	cpuinfo.open("/proc/cpuinfo", ifstream::in);

	//TODO
	/*while(!cpuinfo.eof()) {
		cpuinfo.getline(line,256);


	}*/
}

SerialDeviceInfo::~SerialDeviceInfo() {

}
		
void SerialDeviceInfo::printInfo() {

}

char* SerialDeviceInfo::getName() {
    return NULL;
}