#include <omp.h>
#ifdef CPU

#include "train.h"

void forward_pass(nn_t *nn, double *input, double **A, double **Z){

    int i;

    #pragma omp parallel for private (i) schedule (static)
    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
        
    // RAW dist 1 -> cannot use omp
    for(i = 1; i < nn->n_layers; i++){

        matrix_mul_add(Z[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        matrix_func(A[i], Z[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
        matrix_func(Z[i], Z[i], nn->layers_size[i], 1, nn->dactivation_ptr[i - 1]);
    }
}

double back_prop(nn_t *nn, double *output, double **A, double **Z, double **D, double **d){

    int i, n_l;
    int *l_s;
    double loss;
    double *T;
    double **E, **D_aux;

    n_l = nn->n_layers;
    l_s = nn->layers_size;

    D_aux = alloc_matrix_2v(n_l - 1, &(l_s[1]), &(l_s[0]), init_zero);
    E = alloc_matrix_1v(n_l - 1, &(l_s[1]), init_zero);

    loss = nn->loss(A[n_l - 1], output, l_s[n_l - 1]);

    matrix_sub(E[n_l - 2], A[n_l - 1], output, l_s[n_l - 1], 1);
    matrix_mul_dot(E[n_l - 2], E[n_l - 2], Z[n_l - 1], l_s[n_l - 1], 1);  
    

    T = matrix_transpose(A[n_l - 2], l_s[n_l - 2], 1); 
    matrix_mul(D_aux[n_l - 2], E[n_l - 2], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);
    matrix_free(T);

    matrix_sum(D[n_l - 2], D[n_l - 2], D_aux[n_l - 2], l_s[n_l - 1], l_s[n_l - 2]);
    matrix_sum(d[n_l - 2], d[n_l - 2], E[n_l - 2], l_s[n_l - 1], 1);

    //WAR dist 1??
    #pragma omp parallel for private (i) schedule (static)
    for (i = n_l - 2; i > 0; i--) {
            
        T = matrix_transpose(nn->WH[i], l_s[i + 1], l_s[i]);
        matrix_mul(E[i - 1], T, E[i], l_s[i], l_s[i + 1], l_s[i + 1], 1);
        matrix_free(T);

        matrix_mul_dot(E[i - 1], E[i - 1], Z[i], l_s[i], 1);

        matrix_mul(D_aux[i - 1], E[i - 1], A[i - 1], l_s[i], 1, 1, l_s[i - 1]);

        matrix_sum(D[i - 1], D[i - 1], D_aux[i - 1], l_s[i], l_s[i - 1]);
        matrix_sum(d[i - 1], d[i - 1], E[i - 1], l_s[i], 1);
    }

    matrix_free_2D(D_aux, n_l - 1);
    matrix_free_2D(E, n_l - 1);

    return(loss);

}

void update(nn_t *nn, double **D, double **d, double lr, int batch_size){

    int i;
    #pragma omp parallel for private (i) schedule (static)
    for(i = 0; i < nn->n_layers - 1; i++){

        matrix_mul_cnt(D[i], nn->layers_size[i + 1], nn->layers_size[i],  lr * (1.0 / batch_size));
        matrix_mul_cnt(d[i], nn->layers_size[i + 1], 1,  lr * (1.0 / batch_size));
        matrix_sub(nn->WH[i], nn->WH[i], D[i],  nn->layers_size[i + 1], nn->layers_size[i]);
        matrix_sub(nn->BH[i], nn->BH[i], d[i],  nn->layers_size[i + 1], 1);
        matrix_zero(D[i], nn->layers_size[i + 1], nn->layers_size[i]);
        matrix_zero(d[i], nn->layers_size[i + 1], 1);
    }
}

#endif




#ifdef GPU

#include "train.h"
#include "matrix.cuh"

void forward_pass(nn_t *nn, double *input, double **A, double **Z){

    int i;

    for(i = 0; i < nn->layers_size[0]; i++){
        A[0][i] = input[i];
    }
    
    for(i = 1; i < nn->n_layers; i++){
		cuda_matrix_mul_add<<<blk_in_grid, thr_per_blk>>>(Z[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
		cuda_matrix_func<<<blk_in_grid, thr_per_blk>>>(A[i], Z[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
		cuda_matrix_func<<<blk_in_grid, thr_per_blk>>>(Z[i], Z[i], nn->layers_size[i], 1, nn->dactivation_ptr[i - 1]);
        //cuda_matrix_mul_add(Z[i], nn->WH[i - 1], A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);  
        //cuda_matrix_func(A[i], Z[i], nn->layers_size[i], 1, nn->activation_ptr[i - 1]);
        //cuda_matrix_func(Z[i], Z[i], nn->layers_size[i], 1, nn->dactivation_ptr[i - 1]);
    }
}

double back_prop(nn_t *nn, double *output, double **A, double **Z, double **D, double **d)
{
    int i, n_l;
    int *l_s;
    double loss;
    double *T;
    double **E, **D_aux;
	
	int thr_per_blk, blk_in_grid;
	
	// Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)N / thr_per_blk );

    n_l = nn->n_layers;
    l_s = nn->layers_size;

	D_aux = cuda_alloc_matrix_2v<<<blk_in_grid, thr_per_blk>>>(n_l - 1, &(l_s[1]), &(l_s[0]), init_zero);
	E =  cuda_alloc_matrix_1v<<<blk_in_grid, thr_per_blk>>>(n_l - 1, &(l_s[1]), init_zero);
    //D_aux = cuda_alloc_matrix_2v(n_l - 1, &(l_s[1]), &(l_s[0]), init_zero);
    //E = cuda_alloc_matrix_1v(n_l - 1, &(l_s[1]), init_zero);

    loss = nn->loss(A[n_l - 1], output, l_s[n_l - 1]);

    cuda_matrix_sub<<<blk_in_grid, thr_per_blk>>>(E[n_l - 2], A[n_l - 1], output, l_s[n_l - 1], 1);
    cuda_matrix_mul_dot<<<blk_in_grid, thr_per_blk>>>(E[n_l - 2], E[n_l - 2], Z[n_l - 1], l_s[n_l - 1], 1);
    //cuda_matrix_sub(E[n_l - 2], A[n_l - 1], output, l_s[n_l - 1], 1);
    //cuda_matrix_mul_dot(E[n_l - 2], E[n_l - 2], Z[n_l - 1], l_s[n_l - 1], 1);  
    
    T = cuda_matrix_transpose<<<blk_in_grid, thr_per_blk>>>(A[n_l - 2], l_s[n_l - 2], 1);
    cuda_matrix_mul<<<blk_in_grid, thr_per_blk>>>(D_aux[n_l - 2], E[n_l - 2], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);
    cuda_matrix_free<<<blk_in_grid, thr_per_blk>>>(T);
    //T = cuda_matrix_transpose(A[n_l - 2], l_s[n_l - 2], 1); 
    //cuda_matrix_mul(D_aux[n_l - 2], E[n_l - 2], T, l_s[n_l - 1], 1, 1, l_s[n_l - 2]);
    //cuda_matrix_free(T);

    cuda_matrix_sum<<<blk_in_grid, thr_per_blk>>>(D[n_l - 2], D[n_l - 2], D_aux[n_l - 2], l_s[n_l - 1], l_s[n_l - 2]);
    cuda_matrix_sum<<<blk_in_grid, thr_per_blk>>>(d[n_l - 2], d[n_l - 2], E[n_l - 2], l_s[n_l - 1], 1);
    //cuda_matrix_sum(D[n_l - 2], D[n_l - 2], D_aux[n_l - 2], l_s[n_l - 1], l_s[n_l - 2]);
    //cuda_matrix_sum(d[n_l - 2], d[n_l - 2], E[n_l - 2], l_s[n_l - 1], 1);

    for (i = n_l - 2; i > 0; i--) {
		T = cuda_matrix_transposecuda_vec_prod<<<blk_in_grid, thr_per_blk>>>(nn->WH[i], l_s[i + 1], l_s[i]);
		cuda_matrix_mul<<<blk_in_grid, thr_per_blk>>>(E[i - 1], T, E[i], l_s[i], l_s[i + 1], l_s[i + 1], 1);
		cuda_matrix_free<<<blk_in_grid, thr_per_blk>>>(T);
        //T = cuda_matrix_transpose(nn->WH[i], l_s[i + 1], l_s[i]);
        //cuda_matrix_mul(E[i - 1], T, E[i], l_s[i], l_s[i + 1], l_s[i + 1], 1);
        //cuda_matrix_free(T);

		cuda_matrix_mul_dot<<<blk_in_grid, thr_per_blk>>>(E[i - 1], E[i - 1], Z[i], l_s[i], 1);
        //cuda_matrix_mul_dot(E[i - 1], E[i - 1], Z[i], l_s[i], 1);
		
		cuda_matrix_mul<<<blk_in_grid, thr_per_blk>>>(D_aux[i - 1], E[i - 1], A[i - 1], l_s[i], 1, 1, l_s[i - 1]);
        //cuda_matrix_mul(D_aux[i - 1], E[i - 1], A[i - 1], l_s[i], 1, 1, l_s[i - 1]);
		
		cuda_matrix_sum<<<blk_in_grid, thr_per_blk>>>(D[i - 1], D[i - 1], D_aux[i - 1], l_s[i], l_s[i - 1]);
		cuda_matrix_sum<<<blk_in_grid, thr_per_blk>>>(d[i - 1], d[i - 1], E[i - 1], l_s[i], 1);
        //cuda_matrix_sum(D[i - 1], D[i - 1], D_aux[i - 1], l_s[i], l_s[i - 1]);
        //cuda_matrix_sum(d[i - 1], d[i - 1], E[i - 1], l_s[i], 1);
    }
	cuda_matrix_free_2D<<<blk_in_grid, thr_per_blk>>>(D_aux, n_l - 1);
    cuda_matrix_free_2D<<<blk_in_grid, thr_per_blk>>>(E, n_l - 1);
    //cuda_matrix_free_2D(D_aux, n_l - 1);
    //cuda_matrix_free_2D(E, n_l - 1);

    return(loss);

}

void update(nn_t *nn, double **D, double **d, double lr, int batch_size){

    int i;
	int thr_per_blk, blk_in_grid;
	
	// Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)N / thr_per_blk );
	
    for(i = 0; i < nn->n_layers - 1; i++){
		cuda_matrix_mul_cnt<<<blk_in_grid, thr_per_blk>>>(D[i], nn->layers_size[i + 1], nn->layers_size[i],  lr * (1.0 / batch_size));
		cuda_matrix_mul_cnt<<<blk_in_grid, thr_per_blk>>>(d[i], nn->layers_size[i + 1], 1,  lr * (1.0 / batch_size));
		cuda_matrix_sub<<<blk_in_grid, thr_per_blk>>>(nn->WH[i], nn->WH[i], D[i],  nn->layers_size[i + 1], nn->layers_size[i]);
		cuda_matrix_sub<<<blk_in_grid, thr_per_blk>>>(nn->BH[i], nn->BH[i], d[i],  nn->layers_size[i + 1], 1);
		cuda_matrix_zero<<<blk_in_grid, thr_per_blk>>>(D[i], nn->layers_size[i + 1], nn->layers_size[i]);
		cuda_matrix_zero<<<blk_in_grid, thr_per_blk>>>(d[i], nn->layers_size[i + 1], 1);

        //cuda_matrix_mul_cnt(D[i], nn->layers_size[i + 1], nn->layers_size[i],  lr * (1.0 / batch_size));
        //cuda_matrix_mul_cnt(d[i], nn->layers_size[i + 1], 1,  lr * (1.0 / batch_size));
        //cuda_matrix_sub(nn->WH[i], nn->WH[i], D[i],  nn->layers_size[i + 1], nn->layers_size[i]);
        //cuda_matrix_sub(nn->BH[i], nn->BH[i], d[i],  nn->layers_size[i + 1], 1);
        //cuda_matrix_zero(D[i], nn->layers_size[i + 1], nn->layers_size[i]);
        //cuda_matrix_zero(d[i], nn->layers_size[i + 1], 1);
    }
}

#endif
