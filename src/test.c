#include "test.h"
#include "nn.h"


#ifdef CPU
#include <omp.h>
#include "matrix.h"
    
void forward_pass_test(nn_t *nn, double *input, double **A){

    int i;

    //#pragma omp parallel for private (i) schedule (static) // ganancia insignificante
    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
        

    // RAW dist 1 -> cannot use omp
    for(i = 1; i < nn->n_layers; i++){

        matrix_mul_add(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        matrix_func(A[i], A[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
    }

}



//printf("Expected: %f , Obtained: %f Loss %f\n", output[0], A[nn->n_layers - 1][0], loss);

























#endif


///////////////////////////////////////////////////////////

#ifdef GPU

#include "matrix_gpu.cuh"
#include "globals_gpu.cuh"


void forward_pass_test(nn_t *nn, double *input, double **A){

    int i;
	
	set_kernel_params(); // thr_per_blk and blk_in_grid

    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    
    for(i = 1; i < nn->n_layers; i++){
		cuda_matrix_mul_add(A[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
		cuda_matrix_func(A[i], A[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
    }
}

#endif



// common funs
float precision(int tp, int fp){

    float precision = tp/(tp+fp);

    return(precision);

}

float recall(int tp, int fn){

    float recall = tp/(tp+fn);

    return(recall);

}

float f1(float p, float r){

    float f1 = 2*((p*r)/(p+r));

    return(f1);

}
