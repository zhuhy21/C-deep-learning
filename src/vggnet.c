#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "vggnet.h"
#include "data.h"

void forward_vggnet(vggnet *net)
{
    net->conv1.input=net->input;
    net->conv1.output=(float *)calloc(net->conv1.out_units,sizeof(float));
    conv_forward(&(net->conv1));
    
    net->relu1.output =(float *)calloc(net->relu1.units,sizeof(float));
    net->relu1.input=net->conv1.output;
    relu_forward(&(net->relu1));
    free(net->conv1.output);
    
    net->conv2.output=(float *)calloc(net->conv2.out_units,sizeof(float));
    net->conv2.input=net->relu1.output;
    conv_forward(&(net->conv2));
    free(net->relu1.output);
    
    net->relu2.output =(float *)calloc(net->relu2.units,sizeof(float));
    net->relu2.input=net->conv2.output;
    relu_forward(&(net->relu2));
    free(net->conv2.output);
    
    net->mp1.output =(float *)calloc(net->mp1.out_units,sizeof(float));
    net->mp1.input=net->relu2.output;
    max_pooling_forward(&(net->mp1));
    free(net->relu2.output);
    
    net->conv3.output=(float *)calloc(net->conv3.out_units,sizeof(float));
    net->conv3.input=net->mp1.output;
    conv_forward(&(net->conv3));
    free(net->mp1.output);
    
    
    net->relu3.output =(float *)calloc(net->relu3.units,sizeof(float));
    net->relu3.input=net->conv3.output;
    relu_forward(&(net->relu3));
    free(net->conv3.output);
    
    net->conv4.output=(float *)calloc(net->conv4.out_units,sizeof(float));
    net->conv4.input=net->relu3.output;
    conv_forward(&(net->conv4));
    free(net->relu3.output);
    

    net->relu4.output =(float *)calloc(net->relu4.units,sizeof(float));
    net->relu4.input=net->conv4.output;
    relu_forward(&(net->relu4));
    free(net->conv4.output);

    net->mp2.output =(float *)calloc(net->mp2.out_units,sizeof(float));
    net->mp2.input=net->relu4.output;
    max_pooling_forward(&(net->mp2));
    free(net->relu4.output);
    
    net->conv5.output=(float *)calloc(net->conv5.out_units,sizeof(float));
    net->conv5.input=net->mp2.output;
    conv_forward(&(net->conv5));
    free(net->mp2.output);
    
    net->relu5.output =(float *)calloc(net->relu5.units,sizeof(float));
    net->relu5.input=net->conv5.output;
    relu_forward(&(net->relu5));
    free(net->conv5.output);
    
    net->conv6.output=(float *)calloc(net->conv6.out_units,sizeof(float));
    net->conv6.input=net->relu5.output;
    conv_forward(&(net->conv6));
    free(net->relu5.output);

    net->relu6.output =(float *)calloc(net->relu6.units,sizeof(float));
    net->relu6.input=net->conv6.output;
    relu_forward(&(net->relu6));
    free(net->conv6.output);
    
    net->conv7.output=(float *)calloc(net->conv7.out_units,sizeof(float));
    net->conv7.input=net->relu6.output;
    conv_forward(&(net->conv7));
    
    // for(int i=0;i<7;i++)
    // {
    //     for (int j=0;j<7;j++)
    //     {
    //         printf("%f, ",net->conv7.output[i*7+j]);
    //     }
    //     printf("\n");
    // } 
    // printf("\n");
    // float su=0.0;
    // for(int i=0;i<256;i++)
    // {
    //     printf("%f\n",net->conv7.weights[i*9+4]*net->relu6.output[i*49]);
    // }
    
    free(net->relu6.output);
    

    net->relu7.output =(float *)calloc(net->relu7.units,sizeof(float));
    net->relu7.input=net->conv7.output;
    relu_forward(&(net->relu7));
    free(net->conv7.output);
    
    net->conv8.output=(float *)calloc(net->conv8.out_units,sizeof(float));
    net->conv8.input=net->relu7.output;
    conv_forward(&(net->conv8));
    free(net->relu7.output);
    

    net->relu8.output =(float *)calloc(net->relu8.units,sizeof(float));
    net->relu8.input=net->conv8.output;
    relu_forward(&(net->relu8));
    free(net->conv8.output);
    
    net->mp3.output =(float *)calloc(net->mp3.out_units,sizeof(float));
    net->mp3.input=net->relu8.output;
    max_pooling_forward(&(net->mp3));
    free(net->relu8.output);
    for(int i=0;i<6;i++)
    {
        for (int j=0;j<6;j++)
        {
            printf("%f, ",net->mp3.output[i*6+j]);
        }
        printf("\n");
    }   
    net->conv9.output=(float *)calloc(net->conv9.out_units,sizeof(float));
    net->conv9.input=net->mp3.output;
    conv_forward(&(net->conv9));
    free(net->mp3.output);
    

    net->relu9.output =(float *)calloc(net->relu9.units,sizeof(float));
    net->relu9.input=net->conv9.output;
    relu_forward(&(net->relu9));
    free(net->conv9.output);
    
    net->conv10.output=(float *)calloc(net->conv10.out_units,sizeof(float));
    net->conv10.input=net->relu9.output;
    conv_forward(&(net->conv10));
    free(net->relu9.output);
    

    net->relu10.output =(float *)calloc(net->relu10.units,sizeof(float));
    net->relu10.input=net->conv10.output;
    relu_forward(&(net->relu10));
    free(net->conv10.output);
    
    net->conv11.output=(float *)calloc(net->conv11.out_units,sizeof(float));
    net->conv11.input=net->relu10.output;
    conv_forward(&(net->conv11));
    free(net->relu10.output);
    

    net->relu11.output =(float *)calloc(net->relu11.units,sizeof(float));
    net->relu11.input=net->conv11.output;
    relu_forward(&(net->relu11));
    free(net->conv11.output);
    
    net->conv12.output=(float *)calloc(net->conv12.out_units,sizeof(float));
    net->conv12.input=net->relu11.output;
    conv_forward(&(net->conv12));
    free(net->relu11.output);
    
    
    net->relu12.output =(float *)calloc(net->relu12.units,sizeof(float));
    net->relu12.input=net->conv12.output;
    relu_forward(&(net->relu12));
    free(net->conv12.output);
    
    net->mp4.output =(float *)calloc(net->mp4.out_units,sizeof(float));
    net->mp4.input=net->relu12.output;
    max_pooling_forward(&(net->mp4));
    free(net->relu12.output);
    
    net->conv13.output=(float *)calloc(net->conv13.out_units,sizeof(float));
    net->conv13.input=net->mp4.output;
    conv_forward(&(net->conv13));
    free(net->mp4.output);
    

    net->relu13.output =(float *)calloc(net->relu13.units,sizeof(float));
    net->relu13.input=net->conv13.output;
    relu_forward(&(net->relu13));
    free(net->conv13.output);
    
    net->conv14.output=(float *)calloc(net->conv14.out_units,sizeof(float));
    net->conv14.input=net->relu13.output;
    conv_forward(&(net->conv14));
    free(net->relu13.output);  
    

    net->relu14.output =(float *)calloc(net->relu14.units,sizeof(float));
    net->relu14.input=net->conv14.output;
    relu_forward(&(net->relu14));
    free(net->conv14.output);
    
    net->conv15.output=(float *)calloc(net->conv15.out_units,sizeof(float));
    net->conv15.input=net->relu14.output;
    conv_forward(&(net->conv15));
    free(net->relu14.output);
   

    net->relu15.output =(float *)calloc(net->relu15.units,sizeof(float));
    net->relu15.input=net->conv15.output;
    relu_forward(&(net->relu15));
    free(net->conv15.output);
    
    net->conv16.output=(float *)calloc(net->conv16.out_units,sizeof(float));
    net->conv16.input=net->relu15.output;
    conv_forward(&(net->conv16));
    free(net->relu15.output);
    
    
    net->relu16.output =(float *)calloc(net->relu16.units,sizeof(float));
    net->relu16.input=net->conv16.output;
    relu_forward(&(net->relu16));
    free(net->conv16.output);
    
    net->mp5.output =(float *)calloc(net->mp5.out_units,sizeof(float));
    net->mp5.input=net->relu16.output;
    max_pooling_forward(&(net->mp5));
    free(net->relu16.output);
    
    net->fc1.output = (float *)calloc(net->fc1.out_units,sizeof(float));
    net->fc1.input=net->mp5.output;
    fc_forward(&(net->fc1));
    free(net->mp5.output);
    
    net->relu17.output =(float *)calloc(net->relu6.units,sizeof(float));
    net->relu17.input=net->fc1.output;
    relu_forward(&(net->relu17));
    
    net->fc2.output = (float *)calloc(net->fc2.out_units,sizeof(float));
    net->fc2.input=net->relu17.output;
    fc_forward(&(net->fc2));

    net->relu18.output =(float *)calloc(net->relu18.units,sizeof(float));
    net->relu18.input=net->fc2.output;
    relu_forward(&(net->relu18));

    net->fc3.output = (float *)calloc(net->fc3.out_units,sizeof(float));
    net->fc3.input=net->relu18.output;
    fc_forward(&(net->fc3));

    free(net->fc1.output);
    free(net->fc2.output);
    free(net->relu17.output);
    free(net->relu18.output);
}
void setup_vggnet(vggnet *net, short batchsize)
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

    net->conv2.in_channels = C1_CHANNELS;
    net->conv2.out_channels = C2_CHANNELS;
    net->conv2.in_h = FEATURE1_L;
    net->conv2.in_w = FEATURE1_L;
    net->conv2.kernel_size = C2_KERNEL_L;
    net->conv2.padding = C2_PADDING;
    net->conv2.stride = C2_STRIDES;
    net->conv2.out_h = FEATURE2_L;
    net->conv2.out_w = FEATURE2_L;
    net->conv2.in_units = C1_CHANNELS*FEATURE1_L*FEATURE1_L;
    net->conv2.out_units = C2_CHANNELS*FEATURE2_L*FEATURE2_L;

    net->relu2.units=net->conv2.out_units;

    net->mp1.channels = C2_CHANNELS;
    net->mp1.stride = P1_STRIDES;
    net->mp1.kernel_size = P1_KERNEL_L;
    net->mp1.in_h = FEATURE2_L;
    net->mp1.in_w = FEATURE2_L;
    net->mp1.out_w = POOLING1_L;
    net->mp1.out_h = POOLING1_L;
    net->mp1.in_units = net->relu2.units;
    net->mp1.out_units = C2_CHANNELS*POOLING1_L*POOLING1_L;
    net->mp1.padding=0;

    net->conv3.in_channels = C2_CHANNELS;
    net->conv3.out_channels = C3_CHANNELS;
    net->conv3.in_h = POOLING1_L;
    net->conv3.in_w = POOLING1_L;
    net->conv3.kernel_size = C3_KERNEL_L;
    net->conv3.padding = C3_PADDING;
    net->conv3.stride = C3_STRIDES;
    net->conv3.out_h = FEATURE3_L;
    net->conv3.out_w = FEATURE3_L;
    net->conv3.in_units = C2_CHANNELS*POOLING1_L*POOLING1_L;
    net->conv3.out_units = C3_CHANNELS*FEATURE3_L*FEATURE3_L;

    net->relu3.units=net->conv3.out_units;

    net->conv4.in_channels = C3_CHANNELS;
    net->conv4.out_channels = C4_CHANNELS;
    net->conv4.in_h = FEATURE3_L;
    net->conv4.in_w = FEATURE3_L;
    net->conv4.kernel_size = C4_KERNEL_L;
    net->conv4.padding = C4_PADDING;
    net->conv4.stride = C4_STRIDES;
    net->conv4.out_h = FEATURE4_L;
    net->conv4.out_w = FEATURE4_L;
    net->conv4.in_units = C3_CHANNELS*FEATURE3_L*FEATURE3_L;
    net->conv4.out_units = C4_CHANNELS*FEATURE4_L*FEATURE4_L;

    net->relu4.units=net->conv4.out_units;

    net->mp2.channels = C4_CHANNELS;
    net->mp2.stride = P2_STRIDES;
    net->mp2.kernel_size = P2_KERNEL_L;
    net->mp2.in_h = FEATURE4_L;
    net->mp2.in_w = FEATURE4_L;
    net->mp2.out_w = POOLING2_L;
    net->mp2.out_h = POOLING2_L;
    net->mp2.in_units = net->relu4.units;
    net->mp2.out_units = C4_CHANNELS*POOLING2_L*POOLING2_L;
    net->mp2.padding=0;

    net->conv5.in_channels = C4_CHANNELS;
    net->conv5.out_channels = C5_CHANNELS;
    net->conv5.in_h = POOLING2_L;
    net->conv5.in_w = POOLING2_L;
    net->conv5.kernel_size = C5_KERNEL_L;
    net->conv5.padding = C5_PADDING;
    net->conv5.stride = C5_STRIDES;
    net->conv5.out_h = FEATURE5_L;
    net->conv5.out_w = FEATURE5_L;
    net->conv5.in_units = C4_CHANNELS*POOLING2_L*POOLING2_L;
    net->conv5.out_units = C5_CHANNELS*FEATURE5_L*FEATURE5_L;

    net->relu5.units=net->conv5.out_units;

    net->conv6.in_channels = C5_CHANNELS;
    net->conv6.out_channels = C6_CHANNELS;
    net->conv6.in_h = FEATURE5_L;
    net->conv6.in_w = FEATURE5_L;
    net->conv6.kernel_size = C6_KERNEL_L;
    net->conv6.padding = C6_PADDING;
    net->conv6.stride = C6_STRIDES;
    net->conv6.out_h = FEATURE6_L;
    net->conv6.out_w = FEATURE6_L;
    net->conv6.in_units = C5_CHANNELS*FEATURE5_L*FEATURE5_L;
    net->conv6.out_units = C6_CHANNELS*FEATURE6_L*FEATURE6_L;

    net->relu6.units=net->conv6.out_units;

    net->conv7.in_channels = C6_CHANNELS;
    net->conv7.out_channels = C7_CHANNELS;
    net->conv7.in_h = FEATURE6_L;
    net->conv7.in_w = FEATURE6_L;
    net->conv7.kernel_size = C7_KERNEL_L;
    net->conv7.padding = C7_PADDING;
    net->conv7.stride = C7_STRIDES;
    net->conv7.out_h = FEATURE7_L;
    net->conv7.out_w = FEATURE7_L;
    net->conv7.in_units = C6_CHANNELS*FEATURE6_L*FEATURE6_L;
    net->conv7.out_units = C7_CHANNELS*FEATURE7_L*FEATURE7_L;

    net->relu7.units=net->conv7.out_units;

    net->conv8.in_channels = C7_CHANNELS;
    net->conv8.out_channels = C8_CHANNELS;
    net->conv8.in_h = FEATURE7_L;
    net->conv8.in_w = FEATURE7_L;
    net->conv8.kernel_size = C8_KERNEL_L;
    net->conv8.padding = C8_PADDING;
    net->conv8.stride = C8_STRIDES;
    net->conv8.out_h = FEATURE8_L;
    net->conv8.out_w = FEATURE8_L;
    net->conv8.in_units = C7_CHANNELS*FEATURE7_L*FEATURE7_L;
    net->conv8.out_units = C8_CHANNELS*FEATURE8_L*FEATURE8_L;

    net->relu8.units=net->conv8.out_units;

    net->mp3.channels = C8_CHANNELS;
    net->mp3.stride = P3_STRIDES;
    net->mp3.kernel_size = P3_KERNEL_L;
    net->mp3.in_h = FEATURE8_L;
    net->mp3.in_w = FEATURE8_L;
    net->mp3.out_w = POOLING3_L;
    net->mp3.out_h = POOLING3_L;
    net->mp3.in_units = net->relu8.units;
    net->mp3.out_units = C8_CHANNELS*POOLING3_L*POOLING3_L;
    net->mp3.padding=0;

    net->conv9.in_channels = C8_CHANNELS;
    net->conv9.out_channels = C9_CHANNELS;
    net->conv9.in_h = POOLING3_L;
    net->conv9.in_w = POOLING3_L;
    net->conv9.kernel_size = C9_KERNEL_L;
    net->conv9.padding = C9_PADDING;
    net->conv9.stride = C9_STRIDES;
    net->conv9.out_h = FEATURE9_L;
    net->conv9.out_w = FEATURE9_L;
    net->conv9.in_units = C8_CHANNELS*POOLING3_L*POOLING3_L;
    net->conv9.out_units = C9_CHANNELS*FEATURE9_L*FEATURE9_L;

    net->relu9.units=net->conv9.out_units;

    net->conv10.in_channels = C9_CHANNELS;
    net->conv10.out_channels = C10_CHANNELS;
    net->conv10.in_h = FEATURE9_L;
    net->conv10.in_w = FEATURE9_L;
    net->conv10.kernel_size = C10_KERNEL_L;
    net->conv10.padding = C10_PADDING;
    net->conv10.stride = C10_STRIDES;
    net->conv10.out_h = FEATURE10_L;
    net->conv10.out_w = FEATURE10_L;
    net->conv10.in_units = C9_CHANNELS*FEATURE9_L*FEATURE9_L;
    net->conv10.out_units = C10_CHANNELS*FEATURE10_L*FEATURE10_L;

    net->relu10.units=net->conv10.out_units;

    net->conv11.in_channels = C10_CHANNELS;
    net->conv11.out_channels = C11_CHANNELS;
    net->conv11.in_h = FEATURE10_L;
    net->conv11.in_w = FEATURE10_L;
    net->conv11.kernel_size = C11_KERNEL_L;
    net->conv11.padding = C11_PADDING;
    net->conv11.stride = C11_STRIDES;
    net->conv11.out_h = FEATURE11_L;
    net->conv11.out_w = FEATURE11_L;
    net->conv11.in_units = C10_CHANNELS*FEATURE10_L*FEATURE10_L;
    net->conv11.out_units = C11_CHANNELS*FEATURE11_L*FEATURE11_L;

    net->relu11.units=net->conv11.out_units;

    net->conv12.in_channels = C11_CHANNELS;
    net->conv12.out_channels = C12_CHANNELS;
    net->conv12.in_h = FEATURE11_L;
    net->conv12.in_w = FEATURE11_L;
    net->conv12.kernel_size = C12_KERNEL_L;
    net->conv12.padding = C12_PADDING;
    net->conv12.stride = C12_STRIDES;
    net->conv12.out_h = FEATURE12_L;
    net->conv12.out_w = FEATURE12_L;
    net->conv12.in_units = C11_CHANNELS*FEATURE11_L*FEATURE11_L;
    net->conv12.out_units = C12_CHANNELS*FEATURE12_L*FEATURE12_L;

    net->relu12.units=net->conv12.out_units;

    net->mp4.channels = C12_CHANNELS;
    net->mp4.stride = P4_STRIDES;
    net->mp4.kernel_size = P4_KERNEL_L;
    net->mp4.in_h = FEATURE12_L;
    net->mp4.in_w = FEATURE12_L;
    net->mp4.out_w = POOLING4_L;
    net->mp4.out_h = POOLING4_L;
    net->mp4.in_units = net->relu12.units;
    net->mp4.out_units = C12_CHANNELS*POOLING4_L*POOLING4_L;
    net->mp4.padding=0;

    net->conv13.in_channels = C12_CHANNELS;
    net->conv13.out_channels = C13_CHANNELS;
    net->conv13.in_h = POOLING4_L;
    net->conv13.in_w = POOLING4_L;
    net->conv13.kernel_size = C13_KERNEL_L;
    net->conv13.padding = C13_PADDING;
    net->conv13.stride = C13_STRIDES;
    net->conv13.out_h = FEATURE13_L;
    net->conv13.out_w = FEATURE13_L;
    net->conv13.in_units = C12_CHANNELS*POOLING4_L*POOLING4_L;
    net->conv13.out_units = C13_CHANNELS*FEATURE13_L*FEATURE13_L;

    net->relu13.units=net->conv13.out_units;

    net->conv14.in_channels = C13_CHANNELS;
    net->conv14.out_channels = C14_CHANNELS;
    net->conv14.in_h = FEATURE13_L;
    net->conv14.in_w = FEATURE13_L;
    net->conv14.kernel_size = C14_KERNEL_L;
    net->conv14.padding = C14_PADDING;
    net->conv14.stride = C14_STRIDES;
    net->conv14.out_h = FEATURE14_L;
    net->conv14.out_w = FEATURE14_L;
    net->conv14.in_units = C13_CHANNELS*FEATURE13_L*FEATURE13_L;
    net->conv14.out_units = C14_CHANNELS*FEATURE14_L*FEATURE14_L;

    net->relu14.units=net->conv14.out_units;

    net->conv15.in_channels = C14_CHANNELS;
    net->conv15.out_channels = C15_CHANNELS;
    net->conv15.in_h = FEATURE14_L;
    net->conv15.in_w = FEATURE14_L;
    net->conv15.kernel_size = C15_KERNEL_L;
    net->conv15.padding = C15_PADDING;
    net->conv15.stride = C15_STRIDES;
    net->conv15.out_h = FEATURE15_L;
    net->conv15.out_w = FEATURE15_L;
    net->conv15.in_units = C14_CHANNELS*FEATURE14_L*FEATURE14_L;
    net->conv15.out_units = C15_CHANNELS*FEATURE15_L*FEATURE15_L;

    net->relu15.units=net->conv15.out_units;

    net->conv16.in_channels = C15_CHANNELS;
    net->conv16.out_channels = C16_CHANNELS;
    net->conv16.in_h = FEATURE15_L;
    net->conv16.in_w = FEATURE15_L;
    net->conv16.kernel_size = C16_KERNEL_L;
    net->conv16.padding = C16_PADDING;
    net->conv16.stride = C16_STRIDES;
    net->conv16.out_h = FEATURE16_L;
    net->conv16.out_w = FEATURE16_L;
    net->conv16.in_units = C15_CHANNELS*FEATURE15_L*FEATURE15_L;
    net->conv16.out_units = C16_CHANNELS*FEATURE16_L*FEATURE16_L;

    net->relu16.units=net->conv16.out_units;

    net->mp5.channels = C16_CHANNELS;
    net->mp5.stride = P5_STRIDES;
    net->mp5.kernel_size = P5_KERNEL_L;
    net->mp5.in_h = FEATURE16_L;
    net->mp5.in_w = FEATURE16_L;
    net->mp5.out_w = POOLING5_L;
    net->mp5.out_h = POOLING5_L;
    net->mp5.in_units = net->relu16.units;
    net->mp5.out_units = C16_CHANNELS*POOLING5_L*POOLING5_L;
    net->mp5.padding=0;

    net->fc1.in_units = net->mp5.out_units;
    net->fc1.out_units = FC1_LAYER;
    
    net->relu6.units = FC1_LAYER; 
    
    net->fc2.in_units = FC1_LAYER;
    net->fc2.out_units = FC2_LAYER;

    net->relu7.units = FC2_LAYER;

    net->fc3.in_units = FC2_LAYER;
    net->fc3.out_units = OUT_LAYER;
}
void malloc_vggnet(vggnet *net)
{
    calloc_conv_weights(&(net->conv1));
    calloc_conv_weights(&(net->conv2));
    calloc_conv_weights(&(net->conv3));
    calloc_conv_weights(&(net->conv4));
    calloc_conv_weights(&(net->conv5));
    calloc_conv_weights(&(net->conv6));
    calloc_conv_weights(&(net->conv7));
    calloc_conv_weights(&(net->conv8));
    calloc_conv_weights(&(net->conv9));
    calloc_conv_weights(&(net->conv10));
    calloc_conv_weights(&(net->conv11));
    calloc_conv_weights(&(net->conv12));
    calloc_conv_weights(&(net->conv13));
    calloc_conv_weights(&(net->conv14));
    calloc_conv_weights(&(net->conv15));
    calloc_conv_weights(&(net->conv16));
    calloc_fc_weights(&(net->fc1));
    calloc_fc_weights(&(net->fc2));
    calloc_fc_weights(&(net->fc3));
}
void load_vggnet(vggnet *net, char *filename)
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
    load_conv_weights(&(net->conv6), fp);
    load_conv_weights(&(net->conv7), fp);
    load_conv_weights(&(net->conv8), fp);
    load_conv_weights(&(net->conv9), fp);
    load_conv_weights(&(net->conv10), fp);
    load_conv_weights(&(net->conv11), fp);
    load_conv_weights(&(net->conv12), fp);
    load_conv_weights(&(net->conv13), fp);
    load_conv_weights(&(net->conv14), fp);
    load_conv_weights(&(net->conv15), fp);
    load_conv_weights(&(net->conv16), fp);
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
    static vggnet net;
    setup_vggnet(&net,1);
    malloc_vggnet(&net);
    char weights_path[256]="/share/public/zhuhongyu/AlexNet/pythontry/vggnet.weights";
    load_vggnet(&net,weights_path);
    DataSet dataset;
    load_images(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/t10k-images.idx3-ubyte");
    load_labels(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/t10k-labels.idx1-ubyte");
    int precision=0;
    for(int index=0;index<1;index++)
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
        forward_vggnet(&net);
        int pred=findMaxIndex(net.fc3.output);
        // for(int i=0;i<10;i++)
        // {
        //     printf("%f, ",net.fc3.output[i]);
        // }
        free(net.fc3.output);
        if((int)dataset.labels[index]==pred)
        {
            precision+=1;
        }
        printf("label: %d pred: %d\n", (int)dataset.labels[index],pred);
    }
    //printf("precision=%f\n",precision/1000.0);
}