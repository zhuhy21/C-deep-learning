#include "activation_layer.h"

void relu_forward(nonlinear *op)
{
    for(int i=0;i < op->units; i++)
    {
        if(op->input[i]>0)
            {
                op->output[i]=op->input[i];
            }
        else
        {
            op->output[i]=0.0;
        }
    }
}
