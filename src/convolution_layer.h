#include <stdlib.h>
#include <stdio.h>
typedef struct conv {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    float *input_col;

    int in_channels, out_channels;
    int kernel_size; int padding; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    short batchsize;
} conv;

void conv_forward(conv *op);
void conv_backward(conv *op);

void calloc_conv_weights(conv *op);
void free_conv_weights(conv *op);

void call_conv_dweights(conv *op);
void free_conv_dweights(conv *op);

void load_conv_weights(conv *op, FILE *fp);
void load_conv_weights_nobias(conv *op, FILE *fp);
void save_conv_weights(conv *op, FILE *fp);