#include <stdlib.h>
#include <stdio.h>

typedef struct fc {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    int in_units, out_units;

    short batchsize;
} fc;

void fc_forward(fc *op);
void fc_backward(fc *op);

void calloc_fc_weights(fc *op);
void free_fc_weights(fc *op);

void calloc_fc_dweights(fc *op);
void free_fc_dweights(fc *op);

void load_fc_weights(fc *op, FILE *fp);
void save_fc_weights(fc *op, FILE *fp);