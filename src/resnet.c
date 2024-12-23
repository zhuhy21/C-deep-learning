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
    //OK
    net->conv11.output=(float *)calloc(net->conv11.out_units,sizeof(float));
    net->conv11.input=net->mp1.output;
    conv_forward(&(net->conv11));
    net->relu2.output =(float *)calloc(net->relu2.units,sizeof(float));
    net->relu2.input=net->conv11.output;
    relu_forward(&(net->relu2));
    net->conv12.output=(float *)calloc(net->conv12.out_units,sizeof(float));
    net->conv12.input=net->relu2.output;
    conv_forward(&(net->conv12));

    float *shortcut0=(float *)calloc(net->conv12.out_units,sizeof(float));
    for(int sc=0;sc<net->conv12.out_units;sc++)
    {
        shortcut0[sc]=net->mp1.output[sc]+net->conv12.output[sc];
    }

    net->relu3.output =(float *)calloc(net->relu3.units,sizeof(float));
    net->relu3.input=shortcut0;
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
    free(shortcut1);
    free(shortcut2);
    free(net->conv1.output);
    free(net->conv11.output);
    free(net->conv12.output);
    free(net->conv21.output);
    free(net->conv22.output);
    free(net->conv31.output);
    free(net->conv32.output);
    free(net->convS2.output);
    free(net->convS3.output);
    free(net->relu1.output);
    free(net->relu2.output);
    free(net->relu3.output);
    free(net->relu4.output);
    free(net->relu5.output);
    free(net->relu6.output);
    free(net->relu7.output);
    free(net->ap1.output);
    free(net->mp1.output);
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

    net->conv11.in_channels = C1_CHANNELS;
    net->conv11.out_channels = C11_CHANNELS;
    net->conv11.in_h = POOLING1_L;
    net->conv11.in_w = POOLING1_L;
    net->conv11.kernel_size = C11_KERNEL_L;
    net->conv11.padding = C11_PADDING;
    net->conv11.stride = C11_STRIDES;
    net->conv11.out_h = FEATURE11_L;
    net->conv11.out_w = FEATURE11_L;
    net->conv11.in_units = net->mp1.out_units;
    net->conv11.out_units = C11_CHANNELS*FEATURE11_L*FEATURE11_L;

    net->relu2.units=net->conv11.out_units;

    net->conv12.in_channels = C11_CHANNELS;
    net->conv12.out_channels = C12_CHANNELS;
    net->conv12.in_h = FEATURE11_L;
    net->conv12.in_w = FEATURE11_L;
    net->conv12.kernel_size = C12_KERNEL_L;
    net->conv12.padding = C12_PADDING;
    net->conv12.stride = C12_STRIDES;
    net->conv12.out_h = FEATURE12_L;
    net->conv12.out_w = FEATURE12_L;
    net->conv12.in_units = net->relu2.units;
    net->conv12.out_units = C12_CHANNELS*FEATURE12_L*FEATURE12_L;

    net->relu3.units=net->conv12.out_units;

    net->conv21.in_channels = C12_CHANNELS;
    net->conv21.out_channels = C21_CHANNELS;
    net->conv21.in_h = FEATURE12_L;
    net->conv21.in_w = FEATURE12_L;
    net->conv21.kernel_size = C21_KERNEL_L;
    net->conv21.padding = C21_PADDING;
    net->conv21.stride = C21_STRIDES;
    net->conv21.out_h = FEATURE21_L;
    net->conv21.out_w = FEATURE21_L;
    net->conv21.in_units = net->relu3.units;
    net->conv21.out_units = C21_CHANNELS*FEATURE21_L*FEATURE21_L;

    net->relu4.units=net->conv21.out_units;

    net->conv22.in_channels = C21_CHANNELS;
    net->conv22.out_channels = C22_CHANNELS;
    net->conv22.in_h = FEATURE21_L;
    net->conv22.in_w = FEATURE21_L;
    net->conv22.kernel_size = C22_KERNEL_L;
    net->conv22.padding = C22_PADDING;
    net->conv22.stride = C22_STRIDES;
    net->conv22.out_h = FEATURE22_L;
    net->conv22.out_w = FEATURE22_L;
    net->conv22.in_units = net->relu4.units;
    net->conv22.out_units = C22_CHANNELS*FEATURE22_L*FEATURE22_L;

    net->convS2.in_channels = C12_CHANNELS;
    net->convS2.out_channels = S2_CHANNELS;
    net->convS2.in_h = FEATURE12_L;
    net->convS2.in_w = FEATURE12_L;
    net->convS2.kernel_size = S2_KERNEL_L;
    net->convS2.padding = S2_PADDING;
    net->convS2.stride = S2_STRIDES;
    net->convS2.out_h = FEATURES2_L;
    net->convS2.out_w = FEATURES2_L;
    net->convS2.in_units = net->relu3.units;
    net->convS2.out_units = S2_CHANNELS*FEATURES2_L*FEATURES2_L;

    net->relu5.units=net->conv22.out_units;

    net->conv31.in_channels = C22_CHANNELS;
    net->conv31.out_channels = C31_CHANNELS;
    net->conv31.in_h = FEATURE22_L;
    net->conv31.in_w = FEATURE22_L;
    net->conv31.kernel_size = C31_KERNEL_L;
    net->conv31.padding = C31_PADDING;
    net->conv31.stride = C31_STRIDES;
    net->conv31.out_h = FEATURE31_L;
    net->conv31.out_w = FEATURE31_L;
    net->conv31.in_units = net->relu5.units;
    net->conv31.out_units = C31_CHANNELS*FEATURE31_L*FEATURE31_L;

    net->relu6.units=net->conv31.out_units;

    net->conv32.in_channels = C31_CHANNELS;
    net->conv32.out_channels = C32_CHANNELS;
    net->conv32.in_h = FEATURE31_L;
    net->conv32.in_w = FEATURE31_L;
    net->conv32.kernel_size = C32_KERNEL_L;
    net->conv32.padding = C32_PADDING;
    net->conv32.stride = C32_STRIDES;
    net->conv32.out_h = FEATURE32_L;
    net->conv32.out_w = FEATURE32_L;
    net->conv32.in_units = net->relu6.units;
    net->conv32.out_units = C32_CHANNELS*FEATURE32_L*FEATURE32_L;

    net->convS3.in_channels = C22_CHANNELS;
    net->convS3.out_channels = S3_CHANNELS;
    net->convS3.in_h = FEATURE22_L;
    net->convS3.in_w = FEATURE22_L;
    net->convS3.kernel_size = S3_KERNEL_L;
    net->convS3.padding = S3_PADDING;
    net->convS3.stride = S3_STRIDES;
    net->convS3.out_h = FEATURES3_L;
    net->convS3.out_w = FEATURES3_L;
    net->convS3.in_units = net->relu5.units;
    net->convS3.out_units = S3_CHANNELS*FEATURES3_L*FEATURES3_L;

    net->relu7.units=net->conv32.out_units;

    net->ap1.channels = C32_CHANNELS;
    net->ap1.stride = P2_STRIDES;
    net->ap1.kernel_size = P2_KERNEL_L;
    net->ap1.in_h = FEATURE32_L;
    net->ap1.in_w = FEATURE32_L;
    net->ap1.out_w = POOLING2_L;
    net->ap1.out_h = POOLING2_L;
    net->ap1.in_units = net->relu7.units;
    net->ap1.out_units = C32_CHANNELS*POOLING2_L*POOLING2_L;

    net->fout.in_units = net->ap1.out_units;
    net->fout.out_units = OUT_LAYER;
}
void malloc_resnet(resnet *net)
{
    calloc_conv_weights(&(net->conv1));
    calloc_conv_weights(&(net->conv11));
    calloc_conv_weights(&(net->conv12));
    calloc_conv_weights(&(net->conv21));
    calloc_conv_weights(&(net->conv22));
    calloc_conv_weights(&(net->convS2));
    calloc_conv_weights(&(net->conv31));
    calloc_conv_weights(&(net->conv32));
    calloc_conv_weights(&(net->convS3));
    calloc_fc_weights(&(net->fout));
}
void load_resnet(resnet *net, char *filename)
{
    /**
     * load weights of network from file
     * */
    FILE *fp = fopen(filename, "rb");
    load_conv_weights(&(net->conv1), fp);
    load_conv_weights_nobias(&(net->conv11), fp);
    load_conv_weights_nobias(&(net->conv12), fp);
    load_conv_weights_nobias(&(net->conv21), fp);
    load_conv_weights_nobias(&(net->conv22), fp);
    load_conv_weights(&(net->convS2), fp);
    load_conv_weights_nobias(&(net->conv31), fp);
    load_conv_weights_nobias(&(net->conv32), fp);
    load_conv_weights(&(net->convS3), fp);
    load_fc_weights(&(net->fout), fp);
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
    static resnet net;
    setup_resnet(&net,1);
    malloc_resnet(&net);
    char weights_path[256]="/share/public/zhuhongyu/AlexNet/pythontry/resnet.weights";
    load_resnet(&net,weights_path);
    DataSet dataset;
    load_images(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/t10k-images.idx3-ubyte");
    load_labels(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/t10k-labels.idx1-ubyte");
    int precision=0;
    for(int index=0;index<1000;index++)
    {
        if(index%100==0)
        {
            printf("index:%d\n",index);
        }
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
        forward_resnet(&net);
        int pred=findMaxIndex(net.fout.output);
        free(net.fout.output);
        if((int)dataset.labels[index]==pred)
        {
            precision+=1;
        }
        //printf("label: %d pred: %d\n", (int)dataset.labels[index],pred);
    }
    printf("precision=%f\n",precision/1000.0);
}