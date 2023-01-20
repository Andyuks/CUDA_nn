#ifndef __NN_CUH
#define __NN_CUH

#include "ds.h"
#include "nn.h"

void train(nn_t *nn, ds_t *ds, int epochs, int batches, double lr);

void test(nn_t *nn, ds_t *ds);

#endif