#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ds.h"
#include "nn.h"
#include "nn_aux.h"
#include "utils.h"
#include "train.h"
#include "test.h"
#include "matrix_common.h"

#include "nn_gpu.cuh"
#include "globals_gpu.cuh"
#include "matrix_gpu.cuh"

__global__ void cuda_train_batch(nn_t *nn, ds_t *ds, int size_batch, double lr, 
                                 int n_batches, int *order, double *loss, double **A, double **Z, double **D, double **d)
{
    int min_batch, i;
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    
    if (idx < n_batches) {
        for(min_batch = (idx * size_batch); min_batch < ((idx + 1) * size_batch); min_batch++) {
            i = order[min_batch];
            forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); 
            atomicAdd(loss, back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d));
        }
        _syncthreads();

        if (idx == 0)
            update(nn, D, d, lr, size_batch);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

#ifdef GPU

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr) {
    //int i, n, x, n_batches, min_batch;
    int i, n, n_batches;

    double **A, **Z, **D, **d;
    int *order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
	
  
    order = (int*)malloc(ds->n_samples * sizeof(int));
    
    A = cuda_alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
    Z = cuda_alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
    D = cuda_alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
    d = cuda_alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    
    n_batches = ds->n_samples / size_batch;
    int batch_sample_blks = ceil( (float)n_batches / thr_per_blk );


    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;
    
    for (n=0; n < epochs; n++) {
            
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        loss = 0.0;
        shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);

        /*
        for (x = 0; x < n_batches; x++) {
            for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){
            
                i = order[min_batch];
                forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); 
                loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d);
            }
            
            update(nn, D, d, lr, size_batch);
        }
        */

        cuda_train_batch<<<batch_sample_blks, thr_per_blk>>>(nn, ds, size_batch, lr, n_batches, order, loss, A, Z, D, d);

        clock_gettime(clk_id, &t2);

        if(verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss / ds->n_samples);

    }

}

void test(nn_t *nn, ds_t *ds){
    
    int i;
    double **A;

    A = cuda_alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);

    for(i = 0; i < ds->n_samples; i++){

        forward_pass_test(nn, &ds->inputs[i * ds->n_inputs], A);
    }

    // Precision
    // Recall
    // F1
}

#endif
