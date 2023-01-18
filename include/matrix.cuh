#ifndef __MATRIX_CUH
#define __MATRIX_CUH

double **cuda_alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void));

double **cuda_alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void));

double *cuda_alloc_array(int length);

double *cuda_alloc_matrix(int rows, int cols);

void cuda_matrix_free_2D(double **m, int n_layers);

void cuda_matrix_free(double *m);

__global__ void cuda_matrix_sum(double *c, double *a, double *b, int rows, int cols);

__global__ void cuda_matrix_sub(double *c, double *a, double *b, int rows, int cols);

__global__ void cuda_matrix_mul_cnt(double *m, int rows, int cols, double cnt);

__global__ void cuda_matrix_zero(double *m, int rows, int cols);

__global__ void cuda_matrix_mul_dot(double *c, double *a, double *b, int rows, int cols);

__global__ void cuda_matrix_transpose(double *m, double *m_tr, int rows, int cols);

__global__ void cuda_matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols);

__global__ void cuda_matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double* d);

__global__ void cuda_matrix_func(double *n, double *m, int m_rows, int m_cols, double (*func)(double));

#endif
