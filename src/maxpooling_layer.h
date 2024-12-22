#include <stdlib.h>

typedef struct max_pooling {
    float *input;  float *d_input;
    float *output; float *d_output;
    int channels;
    int kernel_size; int stride;
    int padding;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    short batchsize;
} max_pooling;

void max_pooling_forward(max_pooling *op);
void max_pooling_backward(max_pooling *op);