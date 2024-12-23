#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "convolution_layer.h"
#include "activation_layer.h"
#include "maxpooling_layer.h"
#include "fc_layer.h"

#define IN_CHANNELS 1
#define C1_CHANNELS 64
#define C2_CHANNELS 64
#define C3_CHANNELS 128
#define C4_CHANNELS 128
#define C5_CHANNELS 256
#define C6_CHANNELS 256
#define C7_CHANNELS 256
#define C8_CHANNELS 256
#define C9_CHANNELS 512
#define C10_CHANNELS 512
#define C11_CHANNELS 512
#define C12_CHANNELS 512
#define C13_CHANNELS 512
#define C14_CHANNELS 512
#define C15_CHANNELS 512
#define C16_CHANNELS 512

#define C1_KERNEL_L 3
#define C2_KERNEL_L 3
#define P1_KERNEL_L 2
#define C3_KERNEL_L 3
#define C4_KERNEL_L 3
#define P2_KERNEL_L 2
#define C5_KERNEL_L 3
#define C6_KERNEL_L 3
#define C7_KERNEL_L 3
#define C8_KERNEL_L 3
#define P3_KERNEL_L 2
#define C9_KERNEL_L 3
#define C10_KERNEL_L 3
#define C11_KERNEL_L 3
#define C12_KERNEL_L 3
#define P4_KERNEL_L 2
#define C13_KERNEL_L 3
#define C14_KERNEL_L 3
#define C15_KERNEL_L 3
#define C16_KERNEL_L 3
#define P5_KERNEL_L 2

#define C1_STRIDES 1
#define C2_STRIDES 1
#define P1_STRIDES 2
#define C3_STRIDES 1
#define C4_STRIDES 1
#define P2_STRIDES 2
#define C5_STRIDES 1
#define C6_STRIDES 1
#define C7_STRIDES 1
#define C8_STRIDES 1
#define P3_STRIDES 1
#define C9_STRIDES 1
#define C10_STRIDES 1
#define C11_STRIDES 1
#define C12_STRIDES 1
#define P4_STRIDES 2
#define C13_STRIDES 1
#define C14_STRIDES 1
#define C15_STRIDES 1
#define C16_STRIDES 1
#define P5_STRIDES 1

#define C1_PADDING 1
#define C2_PADDING 1
#define C3_PADDING 1
#define C4_PADDING 1
#define C5_PADDING 1
#define C6_PADDING 1
#define C7_PADDING 1
#define C8_PADDING 1
#define C9_PADDING 1
#define C10_PADDING 1
#define C11_PADDING 1
#define C12_PADDING 1
#define C13_PADDING 1
#define C14_PADDING 1
#define C15_PADDING 1
#define C16_PADDING 1

#define FEATURE0_L 28
#define FEATURE1_L 28
#define FEATURE2_L 28
#define POOLING1_L 14
#define FEATURE3_L 14
#define FEATURE4_L 14
#define POOLING2_L 7
#define FEATURE5_L 7
#define FEATURE6_L 7
#define FEATURE7_L 7
#define FEATURE8_L 7
#define POOLING3_L 6
#define FEATURE9_L 6
#define FEATURE10_L 6
#define FEATURE11_L 6
#define FEATURE12_L 6
#define POOLING4_L 3
#define FEATURE13_L 3
#define FEATURE14_L 3
#define FEATURE15_L 3
#define FEATURE16_L 3
#define POOLING5_L 2

#define FC1_LAYER   4096
#define FC2_LAYER   4096
#define OUT_LAYER   10

typedef struct network {

    float *input;
    float *output;
    short batchsize;

    conv conv1;
    nonlinear relu1;
    conv conv2;
    nonlinear relu2;
    max_pooling mp1;

    conv conv3;
    nonlinear relu3;
    conv conv4;
    nonlinear relu4;
    max_pooling mp2;

    conv conv5;
    nonlinear relu5;
    conv conv6;
    nonlinear relu6;
    conv conv7;
    nonlinear relu7;
    conv conv8;
    nonlinear relu8;
    max_pooling mp3;

    conv conv9;
    nonlinear relu9;
    conv conv10;
    nonlinear relu10;
    conv conv11;
    nonlinear relu11;
    conv conv12;
    nonlinear relu12;
    max_pooling mp4;

    conv conv13;
    nonlinear relu13;
    conv conv14;
    nonlinear relu14;
    conv conv15;
    nonlinear relu15;
    conv conv16;
    nonlinear relu16;
    max_pooling mp5;

    fc fc1;
    nonlinear relu17;
    fc fc2;
    nonlinear relu18;
    fc fc3;
    
}vggnet;

void forward_vggnet(vggnet *net);
void backward_vggnet(vggnet *net, int *batch_Y);