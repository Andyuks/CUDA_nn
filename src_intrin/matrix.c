#include <omp.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nn_aux.h"
#include "globals.h"
#include "matrix.h"
#include "matrix_common.h"

#ifdef TIMING
    #include <time.h>
    #include "utils.h"
#endif


double **alloc_matrix_2v(int n_layers, int *size, int *size_prev, double (*init_weight_ptr)(void)){

    double **m;
    int i, j;

    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * size_prev[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return(NULL);
        }

    for (i = 0; i < n_layers; i++){
        for (j = 0; j < size[i] * size_prev[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    return(m);
}

double **alloc_matrix_1v(int n_layers, int *size, double (*init_weight_ptr)(void)){

    double **m;
    int i, j;

    if ((m = (double**)malloc(n_layers * sizeof(double*))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < n_layers; i++)
        if ((m[i] = (double*)malloc(size[i] * sizeof(double))) == NULL) {
            matrix_free_2D(m, n_layers);
            return(NULL);
        }

    for (i = 0; i < n_layers; i++){
        for (j = 0; j < size[i]; j++){
            m[i][j] = init_weight_ptr();
        }
    }

    return(m);
}

double *alloc_array(int length){

    double *v;
    int i;

    if ((v = (double*)malloc(length* sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < length; i++){
        v[i] = 0.0;
    }
    
    return(v);
}


double *alloc_matrix(int rows, int cols){

    double *m;
    int i;

    if ((m = (double*)malloc(rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    for (i = 0; i < rows * cols; i++){
        m[i] = 0.0;
    }
    
    return(m);
}


void matrix_free_2D(double **m, int n_layers){

    int i;

    for (i = 0; i < n_layers; ++i) {
        if (m[i] != NULL) {
            free(m[i]);
        }
    }
    free(m);
}

void matrix_free(double *m){

    if (m != NULL)
        free(m);
}


void matrix_sum(double *c, double *a, double *b, int rows, int cols){

    int  col, row;
    __m512d va, vb, vc;

    #pragma omp parallel for private (row, col, va, vb, vc) schedule (static)
    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols - cols % 8; col+=8) {
            va = _mm512_loadu_pd(&a[row*cols + col]);
            vb = _mm512_loadu_pd(&b[row*cols + col]);
            vc = _mm512_add_pd(va, vb);
            _mm512_storeu_pd(&c[row*cols + col], vc);
        }
        
        for(; col < cols; col++)
            *m_elem(c, cols, row, col) = *m_elem(a, cols, row, col) + *m_elem(b, cols, row, col);
    }
}

void matrix_sub(double *c, double *a, double *b, int rows, int cols){

    int col, row;
    __m512d va, vb, vc;

    #pragma omp parallel for private (row, col, va, vb, vc) schedule (static)
    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols - cols % 8; col+=8) {
            va = _mm512_loadu_pd(&a[row*cols + col]);
            vb = _mm512_loadu_pd(&b[row*cols + col]);
            vc = _mm512_sub_pd(va, vb);
            _mm512_storeu_pd(&c[row*cols + col], vc);
        }

        for(; col < cols; col++)
            *m_elem(c, cols, row, col) = *m_elem(a, cols, row, col) + *m_elem(b, cols, row, col);
    }
}

void matrix_mul_cnt(double *m, int rows, int cols, double cnt){

    int col, row;
    __m512d v_cnt, vm, vm_d;

    v_cnt = _mm512_set1_pd(cnt);

    #pragma omp parallel for private (row, col, vm, vm_d) shared (v_cnt) schedule (static)
    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols - cols % 8; col+=8) {
            vm = _mm512_loadu_pd(&m[row * cols + col]);
            vm_d = _mm512_mul_pd(v_cnt, vm);
            _mm512_storeu_pd(&m[row * cols + col], vm_d);
        }

        for(; col < cols; col++)
            *m_elem(m, cols, row, col) *= cnt;

    }
}

void matrix_zero(double *m, int rows, int cols){

    int col, row;
    __m512d v0 = _mm512_setzero_pd();

    #pragma omp parallel for private (row, col) shared (v0) schedule (static)
    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols - cols % 8; col+=8) {
            _mm512_storeu_pd(&m[row * cols + col], v0);
        }
        
        for(; col < cols; col++)
           *m_elem(m, cols, row, col) = 0.0;
    }
}


void matrix_mul_dot(double *c, double *a, double *b, int rows, int cols){

    int col, row;
    __m512d va, vb, vc;

    #pragma omp parallel for private (row, col, va, vb, vc) schedule (static)
    for (row = 0; row < rows; row++) {
        for(col = 0; col < cols - cols % 8; col+=8) {
            va = _mm512_loadu_pd(&a[row*cols + col]);
            vb = _mm512_loadu_pd(&b[row*cols + col]);
            vc = _mm512_mul_pd(va, vb);
            _mm512_storeu_pd(&c[row*cols + col], vc);
        }

        for(; col < cols; col++)
            *m_elem(c, cols, row, col) = *m_elem(a, cols, row, col) * *m_elem(b, cols, row, col);
    }
}

double *matrix_transpose(double *m, int rows, int cols){

    double *m_tr;
    int i, j;
    __m512d vm;
    
    if ((m_tr = (double*)malloc(rows * cols * sizeof(double))) == NULL) {
        return(NULL);
    }

    #pragma omp parallel for private (i, j, vm) schedule (static)
    for (i = 0; i < rows; i++){
        for (j = 0; j < cols - cols % 8; j+=8) {
            vm = _mm512_loadu_pd(&m[i*cols + j]);
            _mm512_storeu_pd(&m_tr[j*rows + i], vm);
        }
        
        for(; j < cols; j++)
            *m_elem(m_tr, rows, j, i) = *m_elem(m, cols, i, j);
    }
    
    return(m_tr);
}

void matrix_mul(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols){

    assert(a_cols == b_rows);

    int i, col, row;
    __m512d va, vb, vc;
    __m256i v_idx = _mm256_setr_epi32(0, 1*b_cols, 2*b_cols, 3*b_cols, 4*b_cols, 5*b_cols, 6*b_cols, 7*b_cols); 

#ifdef TIMING
    int res_time;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
    res_time = clock_gettime(clk_id, &t1);
#endif

    #pragma omp parallel for private(row, col, i, va, vb, vc) schedule (static) 
    for (row = 0; row < a_rows; row++) {
        for(col = 0; col < b_cols; col++) {
            vc = _mm512_setzero_pd();

            for (i = 0; i < a_cols - a_cols % 8; i+=8) {
                va = _mm512_loadu_pd(&a[a_cols * row + i]);
                vb = _mm512_i32gather_pd(v_idx, &b[b_cols * i + col], 8);
                vc = _mm512_fmadd_pd(va, vb, vc);
            }

            *m_elem(c, b_cols, row, col) = _mm512_reduce_add_pd(vc);

            for (; i < a_cols; i++)
                 *m_elem(c, b_cols, row, col) += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
        }
    }

#ifdef TIMING
    res_time = clock_gettime(clk_id, &t2);
    printf("Matrix mul execution time: %ld us \n", diff_time(t2, t1));
#endif

}

void matrix_mul_add(double *c, double *a, double *b, int a_rows, int a_cols, int b_rows, int b_cols, double *d){
/*
    assert(a_cols == b_rows);

    int i, col, row;
    __m512d va, vb, vc;
    __m256i v_idx = _mm256_setr_epi32(0, 1*b_cols, 2*b_cols, 3*b_cols, 4*b_cols, 5*b_cols, 6*b_cols, 7*b_cols); 

    #pragma omp parallel for private(row, col, i, va, vb, vc) schedule (static) 
    for (row = 0; row < a_rows; row++) {
        for(col = 0; col < b_cols; col++) {
            vc = _mm512_setzero_pd();

            for (i = 0; i < a_cols - a_cols % 8; i+=8) {
                va = _mm512_loadu_pd(&a[a_cols * row + i]);
                vb = _mm512_i32gather_pd(v_idx, &b[b_cols * i + col], 8);
                vc = _mm512_fmadd_pd(va, vb, vc);
            }

            *m_elem(c, b_cols, row, col) = _mm512_reduce_add_pd(vc) + *m_elem(d, b_cols, row, col);

            for (; i < a_cols; i++)
                 *m_elem(c, b_cols, row, col) += *m_elem(a, a_cols, row, i) * *m_elem(b, b_cols, i, col);
        }
    }
    */
}

void matrix_func(double *n, double *m, int rows, int cols, double (*func)(double)) {
    
    int col, row;
    #pragma omp parallel for private (row, col) schedule (static)
    for(row = 0; row < rows; row++){
        for(col = 0; col < cols; col++){
            *m_elem(n, cols, row, col) = func(*m_elem(m, cols, row, col));
        }
    }
}