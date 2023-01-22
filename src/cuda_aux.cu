#include <cuda.h>
#include <cuda_runtime.h>

void cuda_copyToDev(void *dst, const void *src, size_t count){
    cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

void cuda_copyToHost(void* dst, const void* src, size_t count){
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

void cuda_setDevice(int id){
	cudaSetDevice(id);
}

void cuda_getDeviceCount(int* gpu_n){
	cudaGetDeviceCount(gpu_n);
}