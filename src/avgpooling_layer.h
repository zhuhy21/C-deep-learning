#include <stdlib.h>

typedef struct avg_pooling {
    float *input;  float *d_input;
    float *output; float *d_output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    short batchsize;
} avg_pooling;

void avg_pooling_forward(avg_pooling *op);
void avg_pooling_backward(avg_pooling *op);