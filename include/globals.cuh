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

#endif



#ifndef __GLOBALS_CUH
#define __GLOBALS_CUH

#define THR_PER_BLOCK 1024 // cambiar?
#define gpuErrchk(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#endif