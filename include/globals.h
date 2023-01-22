#ifndef __GLOBALS_H
#define __GLOBALS_H

extern int verbose;
extern int seed;
extern int train_mode;
extern int test_mode;
extern int n_layers;
extern int epochs;
extern int batches;

extern double lr;

extern int *layers;

extern char dataset[100];
extern char scaling[50];
extern char model[100];

#define THR_PER_BLOCK 256 // Best candidate according to Nvidia GPU spec comparison

/*
extern int thr_per_blk;
extern int blk_in_grid;
*/

#endif


/*
#define gpuErrchk(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)



#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif
*/