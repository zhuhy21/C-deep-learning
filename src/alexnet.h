#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "convolution_layer.h"
#include "activation_layer.h"
#include "maxpooling_layer.h"
#include "fc_layer.h"

#define IN_CHANNELS 1
#define C1_CHANNELS 32
#define C2_CHANNELS 64
#define C3_CHANNELS 128
#define C4_CHANNELS 256
#define C5_CHANNELS 256

#define C1_KERNEL_L 3
#define C2_KERNEL_L 3
#define C3_KERNEL_L 3
#define C4_KERNEL_L 3
#define C5_KERNEL_L 3

#define C1_STRIDES 1
#define C2_STRIDES 1
#define C3_STRIDES 1
#define C4_STRIDES 1
#define C5_STRIDES 1

#define C1_PADDING 1
#define C2_PADDING 1
#define C3_PADDING 1
#define C4_PADDING 1
#define C5_PADDING 1

#define FEATURE0_L 28
#define FEATURE1_L 28
#define POOLING1_L 14
#define FEATURE2_L 14
#define POOLING2_L 7
#define FEATURE3_L 7
#define FEATURE4_L 7
#define FEATURE5_L 7
#define POOLING5_L 3

#define FC6_LAYER   1024
#define FC7_LAYER   512
#define OUT_LAYER   10

typedef struct network {

    float *input;
    float *output;
    short batchsize;

    conv conv1;
    nonlinear relu1;
    max_pooling mp1;

    conv conv2;
    nonlinear relu2;
    max_pooling mp2;

    conv conv3;
    nonlinear relu3;

    conv conv4;
    nonlinear relu4;

    conv conv5;
    nonlinear relu5;
    max_pooling mp5;

    fc fc1;
    nonlinear relu6;
    fc fc2;
    nonlinear relu7;
    fc fc3;
}alexnet;

void forward_alexnet(alexnet *net);
void backward_alexnet(alexnet *net, int *batch_Y);