#include "CUDADevice.h"

#include "CUDAIncludes.h"
#include "CUDAUtils.h"
#include "CUDADeviceInfo.h"

CUDADevice::CUDADevice(int device_index): Device(false) {
    m_pDeviceInfo = new CUDADeviceInfo(device_index);
}

__global__ void empty() {
    
}

CUDADevice::~CUDADevice() {

}

void CUDADevice::printInfo() {
    m_pDeviceInfo->printInfo();
}

void CUDADevice::sendData(Bin bin){
    cudaError_t error = cudaMalloc(&m_pOctree, bin.getSize());
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
    error = cudaMemcpy(m_pOctree, bin.getDataPointer(), bin.getSize(), cudaMemcpyHostToDevice);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
}

void CUDADevice::sendHeader(Bin bin) {
    cudaError_t error = cudaMalloc(&m_pHeader, bin.getSize());
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
    error = cudaMemcpy(m_pHeader, bin.getDataPointer(), bin.getSize(), cudaMemcpyHostToDevice);
    if(cudaIsError(error)) {
        cudaPrintError(error);
        exit(1);
    }
}

void CUDADevice::makeFrameBuffer(vector::int2 size){
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
    framebuffer_window window;
    return window;
}

unsigned char* CUDADevice::getFrame() {
    return NULL;
}

unsigned int* CUDADevice::getCosts() {
    return NULL;
}

bool CUDADevice::isCPU(){
    return false;
}
