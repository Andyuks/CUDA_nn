#ifndef __CUDA_AUX_CUH
#define __CUDA_AUX_CUH

void cuda_copyToDev(void* dst, const void* src, size_t count);

void cuda_copyToHost(void* dst, const void* src, size_t count);

void cuda_setDevice(int id);

void cuda_getDeviceCount(int* gpu_n);

#endif