#include "nn.h"

#ifdef CPU
    #include "matrix.h"
#endif
    
#ifdef GPU
	#include "matrix.cuh"
#endif

void cuda_forward_pass_test(nn_t *nn, double *input, double **A){

    int i;
	int thr_per_blk, blk_in_grid;
	
	// Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)N / thr_per_blk );

    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    
    for(i = 1; i < nn->n_layers; i++){
		cuda_matrix_mul_add<<<blk_in_grid, thr_per_blk>>>(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
		cuda_matrix_func<<<blk_in_grid, thr_per_blk>>>(A[i], A[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
        //matrix_mul_add(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        //matrix_func(A[i], A[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
    }
}

