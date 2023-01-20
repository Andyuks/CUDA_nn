#ifndef __TEST_H
#define __TEST_H

#include "nn.h"

void forward_pass_test(nn_t *nn, double *input, double **A);

float precision(int tp, int fp);

float recall(int tp, int fn);

float f1(float p, float r);

#endif
