#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "resnet.h"
#include "data.h"

void forward_resnet(resnet *net)
{
    net->conv1.input=net->input;
    net->conv1.output=(float *)calloc(net->conv1.out_units,sizeof(float));
    conv_forward(&(net->conv1));

    net->relu1.output =(float *)calloc(net->relu1.units,sizeof(float));
    net->relu1.input=net->conv1.output;
    relu_forward(&(net->relu1));

    net->mp1.output =(float *)calloc(net->mp1.out_units,sizeof(float));
    net->mp1.input=net->relu1.output;
    max_pooling_forward(&(net->mp1));

    net->conv11.output=(float *)calloc(net->conv11.out_units,sizeof(float));
    net->conv11.input=net->mp1.output;
    conv_forward(&(net->conv11));
    
    net->relu2.output =(float *)calloc(net->relu2.units,sizeof(float));
    net->relu2.input=net->conv11.output;
    relu_forward(&(net->relu2));

    net->conv12.output=(float *)calloc(net->conv12.out_units,sizeof(float));
    net->conv12.input=net->relu2.output;
    conv_forward(&(net->conv12));

    net->relu3.output =(float *)calloc(net->relu3.units,sizeof(float));
    net->relu3.input=net->conv12.output;
    relu_forward(&(net->relu3));

    net->conv21.output=(float *)calloc(net->conv21.out_units,sizeof(float));
    net->conv21.input=net->relu3.output;
    conv_forward(&(net->conv21));
    
    net->relu4.output =(float *)calloc(net->relu4.units,sizeof(float));
    net->relu4.input=net->conv21.output;
    relu_forward(&(net->relu4));

    net->conv22.output=(float *)calloc(net->conv22.out_units,sizeof(float));
    net->conv22.input=net->relu4.output;
    conv_forward(&(net->conv22));

    net->convS2.output=(float *)calloc(net->convS2.out_units,sizeof(float));
    net->convS2.input=net->relu3.output;
    conv_forward(&(net->convS2));

    float *shortcut1=(float *)calloc(net->conv22.out_units,sizeof(float));
    for(int sc=0;sc<net->conv22.out_units;sc++)
    {
        shortcut1[sc]=net->convS2.output[sc]+net->conv22.output[sc];
    }

    net->relu5.output =(float *)calloc(net->relu5.units,sizeof(float));
    net->relu5.input=shortcut1;
    relu_forward(&(net->relu5));

    net->conv31.output=(float *)calloc(net->conv31.out_units,sizeof(float));
    net->conv31.input=net->relu5.output;
    conv_forward(&(net->conv31));
    
    net->relu6.output =(float *)calloc(net->relu6.units,sizeof(float));
    net->relu6.input=net->conv31.output;
    relu_forward(&(net->relu6));

    net->conv32.output=(float *)calloc(net->conv32.out_units,sizeof(float));
    net->conv32.input=net->relu6.output;
    conv_forward(&(net->conv32));

    net->convS3.output=(float *)calloc(net->convS3.out_units,sizeof(float));
    net->convS3.input=net->relu5.output;
    conv_forward(&(net->convS3));

    float *shortcut2=(float *)calloc(net->conv32.out_units,sizeof(float));
    for(int sc=0;sc<net->conv32.out_units;sc++)
    {
        shortcut2[sc]=net->convS3.output[sc]+net->conv32.output[sc];
    }

    net->relu7.output =(float *)calloc(net->relu7.units,sizeof(float));
    net->relu7.input=shortcut2;
    relu_forward(&(net->relu7));

    net->ap1.output =(float *)calloc(net->ap1.out_units,sizeof(float));
    net->ap1.input=net->relu7.output;
    avg_pooling_forward(&(net->ap1));

    net->fout.output = (float *)calloc(net->fout.out_units,sizeof(float));
    net->fout.input=net->ap1.output;
    fc_forward(&(net->fout));
}
void setup_resnet(resnet *net, short batchsize)
{
    net->conv1.in_channels = IN_CHANNELS;
    net->conv1.out_channels = C1_CHANNELS;
    net->conv1.in_h = FEATURE0_L;
    net->conv1.in_w = FEATURE0_L;
    net->conv1.kernel_size = C1_KERNEL_L;
    net->conv1.padding = C1_PADDING;
    net->conv1.stride = C1_STRIDES;
    net->conv1.out_h = FEATURE1_L;
    net->conv1.out_w = FEATURE1_L;
    net->conv1.in_units = IN_CHANNELS*FEATURE0_L*FEATURE0_L;
    net->conv1.out_units = C1_CHANNELS*FEATURE1_L*FEATURE1_L;

    net->relu1.units=net->conv1.out_units;

    net->mp1.channels = C1_CHANNELS;
    net->mp1.stride = P1_STRIDES;
    net->mp1.kernel_size = P1_KERNEL_L;
    net->mp1.in_h = FEATURE1_L;
    net->mp1.in_w = FEATURE1_L;
    net->mp1.out_w = POOLING1_L;
    net->mp1.out_h = POOLING1_L;
    net->mp1.in_units = net->relu1.units;
    net->mp1.out_units = C1_CHANNELS*POOLING1_L*POOLING1_L;
    net->mp1.padding=P1_PADDING;
}