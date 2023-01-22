#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ds.h"
#include "utils.h"
#include "nn.h" // ambos CPU y GPU
#include "globals.h"

int thr_per_blk, blk_in_grid;

/*
void set_kernel_params() {
    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)batches / thr_per_blk );
}
*/

int main(int argc, char **argv) {

    ds_t ds;
    nn_t nn;
    
    if(argc == 1){
        printf("No arguments passed!\n");
        exit(0);
    }

    parse_arguments(argc, argv);

    /*
    #ifdef GPU
        set_kernel_params();   // thr_per_blk and blk_in_grid
    #endif
    */

    if(train_mode){
        srand(seed);
        read_csv(dataset, &ds, layers[0], layers[n_layers - 1]);
        init_nn(&nn, n_layers, layers);
        train(&nn, &ds, epochs, batches, lr);
        export_nn(&nn, model);
    }
    else if(test_mode){
        import_nn(&nn, model);
        read_csv(dataset, &ds, nn.layers_size[0], nn.layers_size[n_layers - 1]);
        test(&nn, &ds);
    }
    
    return(0);
}

