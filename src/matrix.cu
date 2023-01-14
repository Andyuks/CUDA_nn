/* matrix.c */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "matrix.h"
#include "nn_aux.h"
#include "globals.h"

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif




/* GPU: alloc matrix 2V*/
double **cuda_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void))
{
    double **m;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(gpuErrchk(cudaMalloc(&m, n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    if(idx < n_layers) 
        if (gpuErrchk(cudaMalloc(&m[idx], size[i] * size_prev[i] * sizeof(double))) == NULL) {
            cuda_matrix_free_2D(m, n_layers);
            return(NULL);
        }

    if(idx < (n_layers*size[i] * size_prev[i])) 
		m[idx] = init_weight_ptr();
       
    return(m);
}


/* GPU: alloc matrix 1V*/
double **cuda_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)) {
    double **m;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (gpuErrchk(cudaMalloc(&m, n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    if(idx < n_layers) 
        if (gpuErrchk(cudaMalloc(&m[idx], size[i] * size_prev[i] * sizeof(double))) == NULL) {
            cuda_matrix_free_2D(m, n_layers);
            return(NULL);
        }

    if(idx < (n_layers * size[i])) 
		m[idx] = init_weight_ptr();
       
    return(m);
}


/* GPU: alloc array */
double *cuda_alloc_array(int length) {

    double *v;
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 

    if (gpuErrchk(cudaMalloc(&v, length * sizeof(double))) == NULL) {
        return(NULL);
    }

    if(idx < length)
        v[idx] = 0.0;
    
    return(v);
}




/* GPU: alloc matrix */
double *cuda_alloc_matrix(int rows, int cols) {

    double *m;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gpuErrchk(cudaMalloc(&m, rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    if(idx < (rows * cols))
        m[idx] = 0.0;
    
    return(m);
}



/* GPU: matrix free 2D */
void matrix_free_2D(double **m, int n_layers) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < n_layers)
        if (m[idx] != NULL)
            cudaFree(m[idx]);

    cudaFree(m);
}


/* GPU: matrix free */
void cuda_matrix_free(double *m){

    if (m != NULL)
        cudaFree(m);
}



double *m_elem(double *m, int length, int x, int y){

    return (double*)&m[length * x + y];
}



/* operations */ 

/* GPU: addition of matrix */
__global__ void cuda_matrix_sum (double *C, double *A, double *B, int rows, int cols) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < (rows * cols)) /*ensure threads not outside dim*/
		C[idx] = A[idx] + B[idx];
}



/* GPU: substraction of matrix  */
__global__ void cuda_matrix_sub (double *C, double *A, double *B, int rows, int cols) {
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
__global__ double * cuda_matrix_transpose(double * m, int rows, int cols) {
    double *m_tr;

    if (getErrchk(cudaMalloc(&m_tr, rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }
    
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int i = idx / cols;
	int j = idx % cols;

	m[idx] = m[j * blockDim.x + i];
	//m[idx] = mat[j * rows + i];
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
            for(i = 0; i < THR_PER_BLOCK; i++)
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

    double *c_aux;

    if ((c_aux = (double*)cudaMalloc(a_rows * b_cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    matrix_mul(&c_aux, &A, &B, a_rows, a_cols, b_rows, b_cols);
    matrix_sum(&C, &c_aux, &D, a_rows, b_cols);
}

/*
__global__ void cuda_matrix_mul_add(double *C, double *A, double *B, int a_rows, int a_cols, int b_rows, int b_cols, double *D) {
	assert(a_cols == b_rows);
    double sum = 0;
    int i;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ double tmp[THR_PER_BLOCK];
    
#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif


    if (idx < a_rows*b_cols)
    {
        tmp[threadIdx.x]=A[idx]*B[idx]; // need index inside block 
        __syncthreads();
        if(threadIdx.x == 0)
        {
            for(i=0; i<THR_PER_BLOCK; i++)
                sum += tmp[i];

			sum +=D[idx];
            atomicAdd(C, sum);
        }
    }
	
#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif
}*/



/* GPU:  apply fun to matrix */
__global__ void cuda_matrix_func (double *A, double *B, int rows, int cols, double (*func)(double)) {
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

