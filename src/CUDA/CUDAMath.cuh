#ifndef _CUDA_VECTOR_H
#define _CUDA_VECTOR_H

#include "CUDAIncludes.h"

inline __device__ float3 operator+(const float3 lhs, const float3 rhs){
    return make_float3(lhs.x+rhs.x, lhs.y+rhs.y, lhs.z+rhs.z);
}

inline __device__ float3 operator-(const float3 lhs, const float3 rhs){
    return make_float3(lhs.x-rhs.x, lhs.y-rhs.y, lhs.z-rhs.z);
}

inline __device__ float3 operator-(const float3 v){
    return make_float3(-v.x, -v.y, -v.z);
}

inline __device__ float3 operator*(const float3 lhs, const int rhs){
    return make_float3(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
}

inline __device__ float3 operator*(const float3 lhs, const float rhs){
    return make_float3(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
}

inline __device__ float3 operator/(const float3 lhs, const float3 rhs){
    return make_float3(__fdividef(lhs.x,rhs.x), __fdividef(lhs.y,rhs.y), __fdividef(lhs.z,rhs.z));
}

inline __device__ float3 operator/(const float3 lhs, const float rhs){
    return make_float3(__fdividef(lhs.x,rhs), __fdividef(lhs.y,rhs), __fdividef(lhs.z,rhs));
}

inline __device__ float dot(const float3 lhs, const float3 rhs) {
    return (rhs.x*lhs.x) + (rhs.y*lhs.y) + (rhs.z*lhs.z);
}

inline __device__ float magnitude(const float3 v) {
    return sqrt(dot(v,v));
}

inline __device__ float3 normalize(const float3 v) {
    return v/magnitude(v);
}

inline __device__ float fixed_point_8bit_to_float(const char fixed) {
    const float range =128.0f; // Max value of a 7 bit unsigned integer.
    const float step = 1.0f/range;
    return fixed*step;
}

inline __device__ float min_component(const float3 v) {
    float min = v.x < v.y ? v.x : v.y;
    return min < v.z ? min : v.z;
}

inline __device__ float max_component(float3 v) {
    float min = v.x > v.y ? v.x : v.y;
    return min > v.z ? min : v.z;
}

#endif //_CUDA_VECTOR_H