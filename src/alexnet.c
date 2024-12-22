#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "alexnet.h"
#include "data.h"

void forward_alexnet(alexnet *net)
{
    net->conv1.input = net->input;
    net->conv1.output = (float *)calloc(net->conv1.out_channels*net->conv1.out_h*net->conv1.out_w,sizeof(float));
    conv_forward(&(net->conv1));
    //printf("conv1\n");
    net->relu1.output =(float *)calloc(net->relu1.units,sizeof(float));
    net->relu1.input=net->conv1.output;
    relu_forward(&(net->relu1));
    //printf("relu1\n");
    net->mp1.output =(float *)calloc(net->mp1.out_units,sizeof(float));
    net->mp1.input=net->relu1.output;
    max_pooling_forward(&(net->mp1));
    //printf("mp1\n");
    // for(int i=0;i<5;i++)
    // {
    //     for(int j=0;j<5;j++)
    //     {
    //         printf("%f, ",net->mp1.output[i*14+j]);
    //     }
    //     printf("\n");
    // }
    net->conv2.input = net->mp1.output;
    net->conv2.output = (float *)calloc(net->conv2.out_units,sizeof(float));
    //printf("conv2\n");
    conv_forward(&(net->conv2));
    // for(int i=0;i<5;i++)
    // {
    //     for(int j=0;j<5;j++)
    //     {
    //         printf("%f, ",net->conv2.output[i*14+j]);
    //     }
    //     printf("\n");
    // }
    net->relu2.output =(float *)calloc(net->relu2.units,sizeof(float));
    net->relu2.input=net->conv2.output;
    relu_forward(&(net->relu2));
    //printf("relu2\n");
    net->mp2.output =(float *)calloc(net->mp2.out_units,sizeof(float));
    net->mp2.input=net->relu2.output;
    max_pooling_forward(&(net->mp2));
    // printf("mp2\n");
    // for(int i=0;i<5;i++)
    // {
    //     for(int j=0;j<5;j++)
    //     {
    //         printf("%f, ",net->mp2.output[i*7+j]);
    //     }
    //     printf("\n");
    // }
    // for(int i=0;i<10;i++)
    // {
    // printf("%f\n",net->conv2.bias[i]);
    // }

    net->conv3.input = net->mp2.output;
    net->conv3.output = (float *)calloc(net->conv3.out_units,sizeof(float));
    conv_forward(&(net->conv3));
    //printf("conv3\n");
    net->relu3.output =(float *)calloc(net->relu3.units,sizeof(float));
    net->relu3.input=net->conv3.output;
    relu_forward(&(net->relu3));
    //printf("relu3\n");

    net->conv4.input = net->relu3.output;
    net->conv4.output = (float *)calloc(net->conv4.out_units,sizeof(float));
    conv_forward(&(net->conv4));
    //printf("conv4\n");
    net->relu4.output =(float *)calloc(net->relu4.units,sizeof(float));
    net->relu4.input=net->conv4.output;
    relu_forward(&(net->relu4));
    //printf("relu4\n");

    net->conv5.input = net->relu4.output;
    net->conv5.output = (float *)calloc(net->conv5.out_units,sizeof(float));
    conv_forward(&(net->conv5));
    //printf("conv5\n");
    net->relu5.output =(float *)calloc(net->relu5.units,sizeof(float));
    net->relu5.input=net->conv5.output;
    relu_forward(&(net->relu5));
    //printf("relu5\n");
    net->mp5.output =(float *)calloc(net->mp5.out_units,sizeof(float));
    net->mp5.input=net->relu5.output;
    max_pooling_forward(&(net->mp5));
    //printf("mp5\n");

    net->fc1.output = (float *)calloc(net->fc1.out_units,sizeof(float));
    net->fc1.input=net->mp5.output;
    fc_forward(&(net->fc1));
    //printf("fc1\n");
    net->relu6.output =(float *)calloc(net->relu6.units,sizeof(float));
    net->relu6.input=net->fc1.output;
    relu_forward(&(net->relu6));
    //printf("relu6\n");

    net->fc2.output = (float *)calloc(net->fc2.out_units,sizeof(float));
    net->fc2.input=net->relu6.output;
    fc_forward(&(net->fc2));
    //printf("fc2\n");
    net->relu7.output =(float *)calloc(net->relu7.units,sizeof(float));
    net->relu7.input=net->fc2.output;
    relu_forward(&(net->relu7));
    //printf("relu7\n");

    net->fc3.output = (float *)calloc(net->fc3.out_units,sizeof(float));
    net->fc3.input=net->relu7.output;
    fc_forward(&(net->fc3));
    free(net->conv1.output);
    free(net->conv2.output);
    free(net->conv3.output);
    free(net->conv4.output);
    free(net->conv5.output);
    free(net->relu1.output);
    free(net->relu2.output);
    free(net->relu3.output);
    free(net->relu4.output);
    free(net->relu5.output);
    free(net->relu6.output);
    free(net->relu7.output);
    free(net->mp1.output);
    free(net->mp2.output);
    free(net->mp5.output);
    free(net->fc1.output);
    free(net->fc2.output);
    //printf("fc3\n");
    // for(int i=0;i<net->fc3.out_units;i++)
    // {
    //     printf("%f,",net->fc3.output[i]);
    // }
    // printf("\n");
}
void setup_alexnet(alexnet *net, short batchsize)
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
    

    net->relu1.units=C1_CHANNELS*FEATURE1_L*FEATURE1_L;

    net->mp1.channels = C1_CHANNELS;
    net->mp1.stride = 2;
    net->mp1.kernel_size = 2;
    net->mp1.in_h = FEATURE1_L;
    net->mp1.in_w = FEATURE1_L;
    net->mp1.out_w = POOLING1_L;
    net->mp1.out_h = POOLING1_L;
    net->mp1.in_units = net->relu1.units;
    net->mp1.out_units = C1_CHANNELS*POOLING1_L*POOLING1_L;
    net->mp1.padding=0;

    net->conv2.in_channels = C1_CHANNELS;
    net->conv2.out_channels = C2_CHANNELS;
    net->conv2.in_h = POOLING1_L;
    net->conv2.in_w = POOLING1_L;
    net->conv2.kernel_size = C2_KERNEL_L;
    net->conv2.padding = C2_PADDING;
    net->conv2.stride = C2_STRIDES;
    net->conv2.out_h = FEATURE2_L;
    net->conv2.out_w = FEATURE2_L;
    net->conv2.in_units = net->mp1.out_units;
    net->conv2.out_units = C2_CHANNELS*FEATURE2_L*FEATURE2_L;

    net->relu2.units = net->conv2.out_units;

    net->mp2.channels = C2_CHANNELS;
    net->mp2.stride = 2;
    net->mp2.kernel_size = 2;
    net->mp2.in_h = FEATURE2_L;
    net->mp2.in_w = FEATURE2_L;
    net->mp2.out_w = POOLING2_L;
    net->mp2.out_h = POOLING2_L;
    net->mp2.in_units = net->relu2.units;
    net->mp2.out_units = C2_CHANNELS*POOLING2_L*POOLING2_L;
    net->mp2.padding=0;

    net->conv3.in_channels = C2_CHANNELS;
    net->conv3.out_channels = C3_CHANNELS;
    net->conv3.in_h = POOLING2_L;
    net->conv3.in_w = POOLING2_L;
    net->conv3.kernel_size = C3_KERNEL_L;
    net->conv3.padding = C3_PADDING;
    net->conv3.stride = C3_STRIDES;
    net->conv3.out_h = FEATURE3_L;
    net->conv3.out_w = FEATURE3_L;
    net->conv3.in_units = net->mp2.out_units;
    net->conv3.out_units = C3_CHANNELS*FEATURE3_L*FEATURE3_L;

    net->relu3.units = net->conv3.out_units;

    net->conv4.in_channels = C3_CHANNELS;
    net->conv4.out_channels = C4_CHANNELS;
    net->conv4.in_h = FEATURE3_L;
    net->conv4.in_w = FEATURE3_L;
    net->conv4.kernel_size = C4_KERNEL_L;
    net->conv4.padding = C4_PADDING;
    net->conv4.stride = C4_STRIDES;
    net->conv4.out_h = FEATURE4_L;
    net->conv4.out_w = FEATURE4_L;
    net->conv4.in_units = net->relu3.units;
    net->conv4.out_units = C4_CHANNELS*FEATURE4_L*FEATURE4_L;

    net->relu4.units = net->conv4.out_units;

    net->conv5.in_channels = C4_CHANNELS;
    net->conv5.out_channels = C5_CHANNELS;
    net->conv5.in_h = FEATURE5_L;
    net->conv5.in_w = FEATURE5_L;
    net->conv5.kernel_size = C5_KERNEL_L;
    net->conv5.padding = C5_PADDING;
    net->conv5.stride = C5_STRIDES;
    net->conv5.out_h = FEATURE5_L;
    net->conv5.out_w = FEATURE5_L;
    net->conv5.in_units = net->relu4.units;
    net->conv5.out_units = C5_CHANNELS*FEATURE5_L*FEATURE5_L;

    net->relu5.units = net->conv5.out_units;

    net->mp5.channels = C5_CHANNELS;
    net->mp5.stride = 2;
    net->mp5.kernel_size = 3;
    net->mp5.in_h = FEATURE5_L;
    net->mp5.in_w = FEATURE5_L;
    net->mp5.out_w = POOLING5_L;
    net->mp5.out_h = POOLING5_L;
    net->mp5.in_units = net->relu5.units;
    net->mp5.out_units = C5_CHANNELS*POOLING5_L*POOLING5_L;
    net->mp5.padding=0;

    net->fc1.in_units = net->mp5.out_units;
    net->fc1.out_units = FC6_LAYER;
    
    net->relu6.units = FC6_LAYER; 
    
    net->fc2.in_units = FC6_LAYER;
    net->fc2.out_units = FC7_LAYER;

    net->relu7.units = FC7_LAYER;

    net->fc3.in_units = FC7_LAYER;
    net->fc3.out_units = OUT_LAYER;
}
void alexnet_init_weights(alexnet *net)
{
    net->conv1.weights[0]=1.0;
    net->conv1.weights[1]=2.0;
    net->conv1.weights[2]=1.0;
    net->conv1.weights[3]=2.0;
    net->conv1.weights[4]=2.0;
    net->conv1.weights[5]=-1.0;
    net->conv1.weights[6]=2.0;
    net->conv1.weights[7]=-1.0;
    net->conv1.weights[8]=1.0;
    net->conv1.weights[9]=1.0;
    net->conv1.weights[10]=2.0;
    net->conv1.weights[11]=1.0;
    net->conv1.weights[12]=2.0;
    net->conv1.weights[13]=2.0;
    net->conv1.weights[14]=-1.0;
    net->conv1.weights[15]=2.0;
    net->conv1.weights[16]=-1.0;
    net->conv1.weights[17]=1.0;
    net->fc1.weights[0]=1.0;
    net->fc1.weights[1]=1.0;
    net->fc1.weights[2]=1.0;
    net->fc1.weights[3]=1.0;
    net->fc1.weights[4]=1.0;
    net->fc1.bias[0]=-4.0;
}

void malloc_alexnet(alexnet *net)
{
    calloc_conv_weights(&(net->conv1));
    calloc_conv_weights(&(net->conv2));
    calloc_conv_weights(&(net->conv3));
    calloc_conv_weights(&(net->conv4));
    calloc_conv_weights(&(net->conv5));
    calloc_fc_weights(&(net->fc1));
    calloc_fc_weights(&(net->fc2));
    calloc_fc_weights(&(net->fc3));
}
void load_alexnet(alexnet *net, char *filename)
{
    /**
     * load weights of network from file
     * */
    FILE *fp = fopen(filename, "rb");
    load_conv_weights(&(net->conv1), fp);
    load_conv_weights(&(net->conv2), fp);
    load_conv_weights(&(net->conv3), fp);
    load_conv_weights(&(net->conv4), fp);
    load_conv_weights(&(net->conv5), fp);
    load_fc_weights(&(net->fc1), fp);
    load_fc_weights(&(net->fc2), fp);
    load_fc_weights(&(net->fc3), fp);
    fclose(fp);
    printf("Load weights from \"%s\" successfully... \n", filename);
}
int findMaxIndex(float *arr) {
   int maxIndex = 0; 
    for (int i = 1; i < 10; i++) {
        if (arr[i] > arr[maxIndex]) {
            maxIndex = i; 
        }
    }
    return maxIndex; 
}
int main(int argc, char* argv[])
{
    static alexnet net;
    setup_alexnet(&net,1);
    malloc_alexnet(&net);
    char weights_path[256]="/share/public/zhuhongyu/AlexNet/pythontry/alexnet.weights";
    load_alexnet(&net,weights_path);
    DataSet dataset;
    load_images(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/t10k-images.idx3-ubyte");
    load_labels(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/t10k-labels.idx1-ubyte");
    int precision=0;
    for(int index=0;index<1000;index++)
    {
        DE_GrayImage* image = &dataset.images[index];
        float* image_input = malloc(28*28*sizeof(float));
        for(int i=0;i<28;i++)
        {
            for(int j=0;j<28;j++)
            {
                image_input[i*28+j]= (image->data[i*28+j]/255.0-0.1307)/0.3081;
            }
        }
        net.input=image_input;
        forward_alexnet(&net);
        int pred=findMaxIndex(net.fc3.output);
        free(net.fc3.output);
        if((int)dataset.labels[index]==pred)
        {
            precision+=1;
        }
        //printf("label: %d pred: %d\n", (int)dataset.labels[index],pred);
    }
    printf("precision=%f\n",precision/1000.0);
}