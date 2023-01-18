/* matrix.cu */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nn_aux.h"

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif

#include "matrix.cuh"
#include "globals.cuh"





double **cuda_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)){

    double **m;
    int i, j;

    cudaError_t malloc_call;
    malloc_call = cudaMalloc(&m, n_layers * sizeof(double*));
    
    if (malloc_call != cudaSuccess)
        return NULL;

    for (i = 0; i < n_layers; i++){
	malloc_call = cudaMalloc(&m[i], size[i] * size_prev[i] * sizeof(double));
        if (malloc_call != cudaSuccess) {
            cuda_matrix_free_2D(m, n_layers);
            return NULL;
        }
    }

    for (i = 0; i < n_layers; i++){
        for (j = 0; j < size[i] * size_prev[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    return(m);
}

double **cuda_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)){

    double **m;
    int i, j;

    cudaError_t malloc_call;
    malloc_call = cudaMalloc(&m, n_layers * sizeof(double*));
    
    if (malloc_call != cudaSuccess)
        return NULL;

    for (i = 0; i < n_layers; i++){
        malloc_call = cudaMalloc(&m[idx], size[idx] * sizeof(double));
        if (malloc_call != cudaSuccess) {
            cuda_matrix_free_2D(m, n_layers);
            return NULL;
        }
    }

    for (i = 0; i < n_layers; i++){
        for (j = 0; j < size[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    return(m);
}

double *cuda_alloc_array(int length){

    double *v;
    int i;

    cudaError_t malloc_call;
    malloc_call = cudaMalloc(&v, length * sizeof(double));
    
    if (malloc_call != cudaSuccess)
        return NULL;

    for (i = 0; i < length; i++){
        v[i] = 0.0;
    }
    
    return(v);
}


double *cuda_alloc_matrix(int rows, int cols){

    double *m;
    int i;

    cudaError_t malloc_call;
    malloc_call = cudaMalloc(&m, rows * cols * sizeof(double));
    
    if (malloc_call != cudaSuccess)
        return NULL;

    for (i = 0; i < rows * cols; i++){
        m[i] = 0.0;
    }
    
    return(m);
}


void cuda_matrix_free_2D(double **m, int n_layers){

    int i;

    for (i = 0; i < n_layers; ++i) {
        if (m[i] != NULL) {
            cudaFree(m[i]);
        }
    }
    cudaFree(m);
}

void cuda_matrix_free(double *m){

    if (m != NULL)
        cudaFree(m);
}



double *m_elem(double *m, int length, int x, int y){

    return (double*)&m[length * x + y];
}



/* operations */ 

/* GPU: addition of matrix */
__global__ void cuda_matrix_sum(double *C, double *A, double *B, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		C[idx] = A[idx] + B[idx];
}



/* GPU: substraction of matrix  */
__global__ void cuda_matrix_sub(double *C, double *A, double *B, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		C[idx] = A[idx] - B[idx];
}



/* GPU:  mul cnt  */
__global__ void cuda_matrix_mul_cnt(double *m, int rows, int cols, double cnt) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		m[idx] *= cnt;
}



/* GPU:  zero  */
__global__ void cuda_matrix_zero(double *m, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		m[idx] = 0.0;
}



/* GPU: cuda matrix mul dot  */
__global__ void cuda_matrix_mul_dot(double *C, double *A, double *B, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		C[idx] = A[idx] * B[idx];
}

/* GPU: matrix transpose */
__global__ double * cuda_matrix_transpose(double * m, double * m_tr, int rows, int cols) {

	idx = threadIdx.x + blockIdx.x * blockDim.x;
	i = idx / cols;
	j = idx % cols;

	m_tr[idx] = m[j * blockDim.x + i];
    return(m_tr);
}


/* GPU: cuda matrix mul */
__global__ void cuda_matrix_mul(double *C, double *A, double *B, int a_rows, int a_cols, int b_rows, int b_cols) {
	assert(a_cols == b_rows);
    double sum = 0.0;
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double c_aux[THR_PER_BLOCK];
    
#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif


    if (idx < a_rows * b_cols) {
        c_aux[threadIdx.x] = A[idx] * B[idx]; /* need index inside block */
        __syncthreads();

        if(threadIdx.x == 0) {
            for(i = 0; i < THR_PER_BLOCK; i++) // TODO: mirar si optimizar op cambiando THR_PER_BLOCK
                sum += c_aux[i];
            atomicAdd(C, sum);
        }
    }
	
#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif
}



/* matrix multiplication add */

__global__ void cuda_matrix_mul_add(double *C, double *A, double *B, int a_rows, int a_cols, int b_rows, int b_cols, double *D) {
	assert(a_cols == b_rows);
    double sum = 0.0;
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double c_aux[THR_PER_BLOCK];
    
#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif


    if (idx < a_rows * b_cols) {
        c_aux[threadIdx.x] = A[idx] * B[idx]; /* need index inside block */
        __syncthreads();

        if(threadIdx.x == 0) {
            for(i = 0; i < THR_PER_BLOCK; i++) // TODO: mirar si optimizar op cambiando THR_PER_BLOCK
                sum += c_aux[i];
            sum += D[idx];
            atomicAdd(C, sum);
        }
    }
	
#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif
}




/* GPU:  apply fun to matrix */
__global__ void cuda_matrix_func(double *A, double *B, int rows, int cols, double (*func)(double)) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		A[idx] = func(B[idx]);
}


/* print matrix */
void print_matrix(double *m, int m_rows, int m_cols)
{
    int col, row;
    printf("%d %d\n", m_rows, m_cols);
    for (row = 0; row < m_rows; row++){
        for(col = 0; col < m_cols; col++){
            printf("(%d %d) %.*lf ", row, col, 10, *m_elem(m, m_cols, row, col));
        }
        printf("\n");
    }
    printf("\n");
}

