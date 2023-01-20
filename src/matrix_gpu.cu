/* matrix.cu */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "nn_aux.h"

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif

#include "matrix_gpu.cuh"
#include "globals.h"

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
        malloc_call = cudaMalloc(&m[i], size[i] * sizeof(double));
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



/* operations */ 

/* GPU: addition of matrix */
__global__ void kcuda_matrix_sum(double *C, double *A, double *B, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		C[idx] = A[idx] + B[idx];
}



/* GPU: substraction of matrix  */
__global__ void kcuda_matrix_sub(double *C, double *A, double *B, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		C[idx] = A[idx] - B[idx];
}



/* GPU:  mul cnt  */
__global__ void kcuda_matrix_mul_cnt(double *m, int rows, int cols, double cnt) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		m[idx] *= cnt;
}



/* GPU:  zero  */
__global__ void kcuda_matrix_zero(double *m, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		m[idx] = 0.0;
}



/* GPU: cuda matrix mul dot  */
__global__ void kcuda_matrix_mul_dot(double *C, double *A, double *B, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		C[idx] = A[idx] * B[idx];
}


/* GPU: matrix transpose */
__global__ void kcuda_matrix_transpose(double * m, double * m_tr, int rows, int cols) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = idx / cols;
	int j = idx % cols;

	m_tr[idx] = m[j * blockDim.x + i];
}


/* GPU: cuda matrix mul */
__global__ void kcuda_matrix_mul(double *C, double *A, double *B, int a_rows, int a_cols, int b_rows, int b_cols) {
	assert(a_cols == b_rows);
    double sum = 0.0;
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col = idx / b_cols;
	int row = idx % b_cols;
	
	__shared__ double c_aux[THR_PER_BLOCK];
    
	
#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif


    if (idx < a_rows * b_cols) {
        c_aux[threadIdx.x] = A[idx] * B[col * blockDim.x + row]; /* need index inside block */
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


/* GPU:  apply fun to matrix */
// __global__ void kcuda_matrix_func(double *A, double *B, int rows, int cols, double (*func)(double)) {
// 	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
// 	if(idx < (rows * cols)) /*ensure threads not outside dim*/
// 		A[idx] = func(B[idx]);
// }


__global__ void kcuda_matrix_sigmoid(double *A, double *B, int rows, int cols){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 

    if(idx < (rows * cols)) /*ensure threads not outside dim*/
        A[idx] = 1 / (1 + exp(-B[idx])); 
}


__global__ void kcuda_matrix_dSigmoid(double *A, double *B, int rows, int cols){
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    int sig_z;

    if(idx < (rows * cols)) { /*ensure threads not outside dim*/
        sig_z = 1 / (1 + exp(-B[idx])); 
        A[idx] = sig_z * (1 - sig_z);
    }
}
///////////////////////////////////////////////////////////

/* WRAPPER FUNCTIONS */

void cuda_matrix_sum(double *c, double *a, double *b, int rows, int cols) {
    kcuda_matrix_sum<<<blk_in_grid, thr_per_blk>>>(c, a, b, rows, cols);
}

void cuda_matrix_sub(double *c, double *a, double *b, int rows, int cols){
    kcuda_matrix_sub<<<blk_in_grid, thr_per_blk>>>(c, a, b, rows, cols);
}

void cuda_matrix_mul_cnt(double *m, int rows, int cols, double cnt){
    kcuda_matrix_mul_cnt<<<blk_in_grid, thr_per_blk>>>(m, rows, cols, cnt);
}

void cuda_matrix_zero(double *m, int rows, int cols){
    kcuda_matrix_zero<<<blk_in_grid, thr_per_blk>>>(m, rows, cols);
}

void cuda_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){
    kcuda_matrix_mul_dot<<<blk_in_grid, thr_per_blk>>>(c, a, b, rows, cols);
}

double * cuda_matrix_transpose(double *m, int rows, int cols){
    double *m_tr;
    m_tr = cuda_alloc_matrix(rows, cols);
    kcuda_matrix_transpose<<<blk_in_grid, thr_per_blk>>>(m, m_tr, rows, cols);
    return m_tr;
}

void cuda_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){
    kcuda_matrix_mul<<<blk_in_grid, thr_per_blk>>>(c, a, b, a_rows, a_cols, b_rows, b_cols);
}

void cuda_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double* d){
    double *c_aux;
    c_aux = cuda_alloc_matrix(a_rows, b_cols);
    kcuda_matrix_mul<<<blk_in_grid, thr_per_blk>>>(c_aux, a, b, a_rows, a_cols, b_rows, b_cols);
    kcuda_matrix_sum<<<blk_in_grid, thr_per_blk>>>(c, c_aux, d, a_rows, b_cols);
    cuda_matrix_free(c_aux);
}

void cuda_matrix_sigmoid(double *n, double *m, int m_rows, int m_cols){
    //kcuda_matrix_func<<<blk_in_grid, thr_per_blk>>>(n, m, m_rows, m_cols, func);
    kcuda_matrix_sigmoid<<<blk_in_grid, thr_per_blk>>>(n, m, m_rows, m_cols);
}

void cuda_matrix_dSigmoid(double *n, double *m, int m_rows, int m_cols){
    //kcuda_matrix_func<<<blk_in_grid, thr_per_blk>>>(n, m, m_rows, m_cols, func);
    kcuda_matrix_dSigmoid<<<blk_in_grid, thr_per_blk>>>(n, m, m_rows, m_cols);
}
