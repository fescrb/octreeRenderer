#include "CUDADevice.h"

CUDADevice::CUDADevice(): Device(false) {

}

__global__ void empty() {
    
}

CUDADevice::~CUDADevice() {

}

void CUDADevice::printInfo() {

}

void CUDADevice::sendData(Bin bin){

}

void CUDADevice::sendHeader(Bin bin) {

}

void CUDADevice::makeFrameBuffer(int2 size){
    Device::makeFrameBuffer(size);
}

void CUDADevice::setRenderInfo(renderinfo* info) {

}

void CUDADevice::advanceTask(int index) {

}

void CUDADevice::renderTask(int index) {
    empty<<<1,1>>>();
}

void CUDADevice::calculateCostsForTask(int index) {

}

framebuffer_window CUDADevice::getFrameBuffer() {

}

unsigned char* CUDADevice::getFrame() {

}

unsigned int* CUDADevice::getCosts() {

}

bool CUDADevice::isCPU(){
    return false;
}
