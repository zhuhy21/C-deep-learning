#include <stdlib.h>
#include <stdio.h>

typedef struct nonlinear {
    float *input; float *d_input;
    float *output; float *d_output;
    int units;

    short batchsize;
} nonlinear;

void relu_forward(nonlinear *op);
void relu_backward(nonlinear *op);

void call_nonlinear_weights(nonlinear *op);
void free_nonlinear_weights(nonlinear *op);