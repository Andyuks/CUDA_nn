#include <stdio.h>
#include <stdlib.h>
#include "globals.h"
#include "matrix_common.h"

double *m_elem(double *m, int length, int x, int y){

    return (double*)&m[length * x + y];
}


void print_matrix(double *m, int m_rows, int m_cols){
    
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
