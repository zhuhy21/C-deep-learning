#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "convolution_layer.h"
#include "activation_layer.h"
#include "maxpooling_layer.h"
#include "fc_layer.h"
#include "avgpooling_layer.h"

#define IN_CHANNELS 1
#define C1_CHANNELS 16
#define C11_CHANNELS 16
#define C12_CHANNELS 16
#define C21_CHANNELS 32
#define C22_CHANNELS 32
#define S2_CHANNELS 32
#define C31_CHANNELS 64
#define C32_CHANNELS 64
#define S3_CHANNELS 64

#define C1_KERNEL_L 3
#define P1_KERNEL_L 3
#define C11_KERNEL_L 3
#define C12_KERNEL_L 3
#define C21_KERNEL_L 3
#define C22_KERNEL_L 3
#define S2_KERNEL_L 3
#define C31_KERNEL_L 3
#define C32_KERNEL_L 3
#define S3_KERNEL_L 3
#define P2_KERNEL_L 7

#define C1_PADDING 1
#define P1_PADDING 1
#define C11_PADDING 1
#define C12_PADDING 1
#define C21_PADDING 1
#define C22_PADDING 1
#define S2_PADDING 1
#define C31_PADDING 1
#define C32_PADDING 1
#define S3_PADDING 1

#define C1_STRIDES 1
#define P1_STRIDES 1
#define C11_STRIDES 1
#define C12_STRIDES 1
#define C21_STRIDES 2
#define C22_STRIDES 1
#define S2_STRIDES 2
#define C31_STRIDES 2
#define C32_STRIDES 1
#define S3_STRIDES 2
#define P2_STRIDES 1

#define FEATURE0_L 28
#define FEATURE1_L 28
#define POOLING1_L 28
#define FEATURE11_L 28
#define FEATURE12_L 28
#define FEATURE21_L 14
#define FEATURE22_L 14
#define FEATURES2_L 14
#define FEATURE31_L 7
#define FEATURE32_L 7
#define FEATURES3_L 7
#define POOLING2_L 1

#define OUT_LAYER   10

typedef struct network {

    float *input;
    float *output;
    short batchsize;

    conv conv1;
    nonlinear relu1;
    max_pooling mp1;

    conv conv11;
    nonlinear relu2;
    conv conv12;
    nonlinear relu3;

    conv conv21;
    nonlinear relu4;
    conv conv22;
    conv convS2;
    nonlinear relu5;

    conv conv31;
    nonlinear relu6;
    conv conv32;
    conv convS3;
    nonlinear relu7;

    avg_pooling ap1;
    fc fout;
}resnet;

void forward_resnet(resnet *net);
void backward_resnet(resnet *net, int *batch_Y);