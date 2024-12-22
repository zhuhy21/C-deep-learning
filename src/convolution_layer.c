#include "convolution_layer.h"

void conv_forward(conv *op)
{
    int half_kernel=op->kernel_size/2;
    for(int out_c=0;  out_c < op->out_channels;out_c++)
    {
        for(int out_y=0;  out_y < op->out_h;out_y++)
        {
            for(int out_x=0;  out_x < op->out_w; out_x++)
            {
                float conv_sum=0.0;
                for(int in_c=0;  in_c < op->in_channels;in_c++)
                {
                    for(int ky=0;ky<op->kernel_size;ky++)
                    {
                        for(int kx=0;kx<op->kernel_size;kx++)
                        {
                            int inY = out_y * op->stride + ky - op->padding;
                            int inX = out_x * op->stride + kx - op->padding;
                            if (inY >= 0 && inY < op->in_h && inX >= 0 && inX < op->in_w) {
                                float inputVal = op->input[in_c*op->in_h*op->in_w + inY * op->in_w + inX];
                                float kernelVal = op->weights[out_c*op->in_channels*op->kernel_size*op->kernel_size+ in_c*op->kernel_size*op->kernel_size+ky * op->kernel_size + kx ];
                                //printf("%f   ", conv_sum);
                                conv_sum += inputVal * kernelVal;
                                //printf("%d,%d,%d,%d,%d,%d\n",out_c,out_y,out_x,in_c,ky,kx);
                                //printf("%f,%f+=%f\n",inputVal,kernelVal,conv_sum);
                            }
                        }
                    }
                }
                op->output[out_c*op->out_w*op->out_h+out_y*op->out_w+out_x]=op->bias[out_c]+conv_sum;
            }
        }
    }
}

void conv_backward(conv *op)
{

}

void calloc_conv_weights(conv *op)
{
    op->weights = (float *)calloc(op->out_channels * op->in_channels * op->kernel_size * op->kernel_size,sizeof(float));
    op->bias    = (float *)calloc(op->out_channels,sizeof(float));
}

void free_conv_weights(conv *op)
{
    free(op->weights);
    free(op->bias);
}

void load_conv_weights(conv *op, FILE *fp)
{
    fread(op->weights, sizeof(float), op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, fp);
    fread(op->bias,    sizeof(float), op->out_channels, fp);
}
void load_conv_weights_nobias(conv *op, FILE *fp)
{
    fread(op->weights, sizeof(float), op->out_channels * op->in_channels * op->kernel_size * op->kernel_size, fp);
}
// int main()
// {
//     conv conv1;
//     conv1.in_channels = 3;
//     conv1.out_channels = 6;
//     conv1.in_h = 3;
//     conv1.in_w = 3;
//     conv1.kernel_size = 3;
//     conv1.padding = 1;
//     conv1.stride = 1;
//     conv1.out_h = 3;
//     conv1.out_w = 3;
//     conv1.weights=[]
// }
