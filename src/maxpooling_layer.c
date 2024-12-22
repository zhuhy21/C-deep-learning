#include "maxpooling_layer.h"

void max_pooling_forward(max_pooling *op)
{
    for(int c=0;c < op->channels; c++)
    {
        for(int h=0;h < op->out_h; h++)
        {
            for(int w=0;w < op->out_w; w++)
            {
                int inY=h*op->stride-op->padding;
                int inX=w*op->stride-op->padding;
                float max=-1.0;
                for(int ky=0;ky < op->kernel_size;ky++)
                {
                    for(int kx=0;kx < op->kernel_size; kx++)
                    {
                        if ((inY+ky) >= 0 && (inY+ky) < op->in_h && (inX+kx) >= 0 && (inX+kx) < op->in_w){
                        if(op->input[c*op->in_h*op->in_w+(inY+ky)*op->in_w+(inX+kx)] > max)
                        {
                            max=op->input[c*op->in_h*op->in_w+(inY+ky)*op->in_w+(inX+kx)];
                        }
                        }
                    }
                }
                op->output[c*op->out_h*op->out_w+h*op->out_w+w]=max;
            }
        }
    }
}