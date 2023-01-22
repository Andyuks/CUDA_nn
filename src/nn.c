#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <float.h>

#include "ds.h"
#include "nn.h"
#include "nn_aux.h"
#include "utils.h"
#include "train.h"
#include "test.h"
#include "matrix_common.h"
#include "globals.h"

#include "matrix.h"


#ifdef GPU
#include "matrix_gpu.cuh"
#include "cuda_aux.cuh"
#endif

void init_nn(nn_t *nn, int n_layers, int *layers_size) {

    int i;

    nn->n_layers = n_layers;
    nn->layers_size = layers_size;
    nn->init_weight_ptr = init_weight_rnd;
    nn->activation_ptr= (activation_ptr_t*)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    nn->dactivation_ptr= (activation_ptr_t*)malloc((nn->n_layers - 1) * sizeof(activation_ptr_t));
    for(i = 0; i < n_layers - 1; i++){
        nn->activation_ptr[i] = sigmoid;
        nn->dactivation_ptr[i] = dSigmoid;
    }
    nn->loss = mse;

    #ifdef CPU
    nn->BH = alloc_matrix_1v(n_layers - 1, &layers_size[1], nn->init_weight_ptr);
    nn->WH = alloc_matrix_2v(n_layers - 1, &layers_size[1], &layers_size[0], nn->init_weight_ptr);
    #endif

    #ifdef GPU
    nn->BH = cuda_alloc_matrix_1v(n_layers - 1, &layers_size[1], nn->init_weight_ptr);
    nn->WH = cuda_alloc_matrix_2v(n_layers - 1, &layers_size[1], &layers_size[0], nn->init_weight_ptr);
    #endif
    
}

//////////////////////////////////////////////////////////////////////////////////////////

#ifdef CPU

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr){

    int i, n, x, n_batches, min_batch;
    double **A, **Z, **D, **d;;
    int *order;
    double loss;
    struct timespec t1, t2;
    clockid_t clk_id = CLOCK_MONOTONIC;
  
    order = (int*)malloc(ds->n_samples * sizeof(int));
    
    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
    Z = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero); 
    D = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]), &(nn->layers_size[0]), init_zero);
    d = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    
    n_batches = ds->n_samples / size_batch;

    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;
    
    for (n=0; n < epochs;n++) {
            
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        loss = 0.0;
        shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);


        #pragma omp parallel for private (x, min_batch, i) shared (A, Z, D, d) reduction (+:loss) schedule (static)
            for (x = 0; x < n_batches; x++) {
                for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){
                
                    i = order[min_batch];
                    forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); 
                    loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d);
                }
            }

        for (x = 0; x < n_batches; x++)
            update(nn, D, d, lr, size_batch);


        clock_gettime(clk_id, &t2);

        if(verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss / ds->n_samples);

    }

}

void test(nn_t *nn, ds_t *ds){
    
    int i;
    double **A;

    A = alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);

    #pragma omp parallel for private (i) schedule (static)
    for(i = 0; i < ds->n_samples; i++){
        forward_pass_test(nn, &ds->inputs[i * ds->n_inputs], A);
    }

    result_management(&ds->outputs[(ds->n_samples-1) * ds->n_outputs], A, nn->layers_size[nn->n_layers - 1], nn->n_layers - 1);
}

#endif


//////////////////////////////////////////////////////////////////////////////////////////

#ifdef GPU

void train(nn_t *nn, ds_t *ds, int epochs, int size_batch, double lr) {
    int i, n, x, n_batches, min_batch;

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
    //int batch_sample_blks = ceil( (float)n_batches / thr_per_blk );

    for(i = 0; i < ds->n_samples; i++)
        order[i] = i;


    for (n=0; n < epochs; n++) {
            
        if(verbose)
            printf("Epoch %d/%d \n", n, epochs);
        
        loss = 0.0;
        shuffle(order, ds->n_samples);

        clock_gettime(clk_id, &t1);


        
        // #pragma omp parallel for private (x, min_batch, i) shared (A, Z, D, d) reduction (+:loss) schedule (static)
            for (x = 0; x < n_batches; x++) {
                for(min_batch = (x * size_batch); min_batch < ((x + 1) * size_batch); min_batch++){
                
                    i = order[min_batch];
                    forward_pass(nn, &ds->inputs[i * ds->n_inputs], A, Z); 
                    loss += back_prop(nn, &ds->outputs[i * ds->n_outputs], A, Z, D, d);
                }

                update(nn, D, d, lr, size_batch);
            }

        /*
        for (x = 0; x < n_batches; x++)
            update(nn, D, d, lr, size_batch);
        */
                    
        // cuda_train_batch<<<batch_sample_blks, thr_per_blk>>>(nn, ds, size_batch, lr, n_batches, order, loss, A, Z, D, d);

        clock_gettime(clk_id, &t2);

        if(verbose)
            printf(" time: %ld us - loss: %.*f\n", diff_time(t2, t1), 12, loss / ds->n_samples);

    }

}

void test(nn_t *nn, ds_t *ds) {
    
    int i;
    double **A;

    A = cuda_alloc_matrix_1v(nn->n_layers, nn->layers_size, init_zero);

    for(i = 0; i < ds->n_samples; i++){
        forward_pass_test(nn, &ds->inputs[i * ds->n_inputs], A);
    }

    result_management(&ds->outputs[(ds->n_samples-1) * ds->n_outputs], A, nn->layers_size[nn->n_layers - 1], nn->n_layers - 1);
}

#endif


//////////////////////////////////////////////////////////////////////////////////////////

void result_management(double * output, double **A, int length, int layer) {
    int i;
    unsigned int tp, tn, fp, fn;
    tp = tn = fp = fn = 0;

    float precision_out, recall_out, f1_out;

    for(i = 0; i < length; i++){
        printf ("A[%d][%d] = %f ; output[%d] = %f \n", layer, i, A[layer][i], i, output[i]);
        if      (A[layer][i] >= 0.5  &&  output[i] == 1)   tp++;
        else if (A[layer][i] >= 0.5  &&  output[i] == 0)   fp++;
        else if (A[layer][i] < 0.5   &&  output[i] == 1)   fn++;
        else if (A[layer][i] < 0.5   &&  output[i] == 0)   tn++;
    }

    printf(" True Positives = %u, True Negatives = %u\n False Positives = %u, False Negatives = %u \n\n", tp, tn, fp, fn);
    precision_out = precision(tp, fp);      // Precision
    recall_out = recall(tp, fn);            // Recall
    f1_out = f1(precision_out, recall_out); // F1

    if (precision_out == FLT_MIN)
        printf("Precision: --- \n");
    else 
        printf("Precision: %f \n", precision_out);

    if (recall_out == FLT_MIN)
        printf("Recall: --- \n");
    else 
        printf("Recall: %f \n", recall_out);
    
    if (f1_out == FLT_MIN)
        printf("F1 (F-score): --- \n");
    else 
        printf("F1 (F-score): %f \n", f1_out);
}


void print_nn(nn_t *nn){

    int i, j, k;
    
    printf("Layers (I/H/O)\n");

    for (i = 0; i < nn->n_layers; i++) {
        printf("%d ", nn->layers_size[i]);
    }
    printf("\n");
    
    printf("Hidden Biases\n ");

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            printf("%lf ", nn->BH[i][j]);
        }
        printf("\n");
    }

    printf("Hidden Weights\n ");
    
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                printf("%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            printf("\n");
        }
    }

}

void import_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"r")) == NULL){
        perror("Error importing the model\n");
        exit(1);
    }
    
    fscanf(fd, "%d ", &n_layers);

    layers = (int*)malloc(n_layers * sizeof(int));

    for (i = 0; i < n_layers; i++) {
        fscanf(fd, "%d ", &(layers[i]));
    }

    init_nn(nn, n_layers, layers);
    
    #ifdef CPU
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fscanf(fd, "%lf ", &(nn->BH[i][j]));
        }
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fscanf(fd, "%lf ", &(nn->WH[i][(j * nn->layers_size[i]) + k]));
            }
        }
    }
    #endif

    #ifdef GPU
    
    double **BH_h = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    double **WH_h = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]),  &(nn->layers_size[0]), init_zero);

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fscanf(fd, "%lf ", &(BH_h[i][j]));
        }
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fscanf(fd, "%lf ", &(WH_h[i][(j * nn->layers_size[i]) + k]));
            }
        }
    }

    cuda_copyToDev(nn->BH, BH_h, (nn->n_layers - 1) * (nn->layers_size[1]) * sizeof(double));
    cuda_copyToDev(nn->WH, WH_h, (nn->n_layers - 1) * (nn->layers_size[1]) * (nn->layers_size[0]) * sizeof(double));

    matrix_free_2D(BH_h, nn->n_layers - 1);
    matrix_free_2D(WH_h, nn->n_layers - 1);


    #endif

    fclose(fd);
}

void export_nn(nn_t *nn, char *filename){

    int i, j, k;
    FILE *fd;

    if ((fd = fopen(filename,"w")) == NULL){
        perror("Error exporting the model");
        exit(1);
    }
    
    fprintf(fd, "%d\n", nn->n_layers);

    for (i = 0; i < nn->n_layers; i++) {
        fprintf(fd, "%d ", nn->layers_size[i]);
    }
    fprintf(fd, "\n");
    
    #ifdef CPU
    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fprintf(fd, "%lf ", nn->BH[i][j]);
        }
        fprintf(fd, "\n");
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fprintf(fd, "%lf ", nn->WH[i][(j * nn->layers_size[i]) + k]);
            }
            fprintf(fd, "\n");
        }
    }
    #endif

    #ifdef GPU
    double **BH_h = alloc_matrix_1v(nn->n_layers - 1, &(nn->layers_size[1]), init_zero);
    double **WH_h = alloc_matrix_2v(nn->n_layers - 1, &(nn->layers_size[1]),  &(nn->layers_size[0]), init_zero);

    cuda_copyToHost(BH_h, nn->BH, (nn->n_layers - 1) * (nn->layers_size[1]) * sizeof(double));
    cuda_copyToHost(WH_h, nn->WH, (nn->n_layers - 1) * (nn->layers_size[1]) * (nn->layers_size[0]) * sizeof(double));

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            fprintf(fd, "%lf ", BH_h[i][j]);
        }
        fprintf(fd, "\n");
    }

    for (i = 0; i < nn->n_layers - 1; i++) {
        for (j = 0; j < nn->layers_size[i + 1]; j++) {
            for(k = 0; k < nn->layers_size[i]; k++) {
                fprintf(fd, "%lf ", WH_h[i][(j * nn->layers_size[i]) + k]);
            }
            fprintf(fd, "\n");
        }
    }

    matrix_free_2D(BH_h, nn->n_layers - 1);
    matrix_free_2D(WH_h, nn->n_layers - 1);

    #endif
    fclose(fd);
}
