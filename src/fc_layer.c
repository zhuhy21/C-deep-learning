#include "fc_layer.h"

void fc_forward(fc *op)
{
    for(int i=0;i<op->out_units;i++)
    {
        float sum=0.0;
        for(int j=0;j<op->in_units;j++)
        {
            sum+=op->weights[i*op->in_units+j]*op->input[j];
        }
        op->output[i]=op->bias[i]+sum;
    }
}

void calloc_fc_weights(fc *op)
{
    op->weights = (float *)calloc(op->in_units * op->out_units, sizeof(float));
    op->bias    = (float *)calloc(op->out_units, sizeof(float));
}
void load_fc_weights(fc *op, FILE *fp)
{
    fread(op->weights, sizeof(float), op->in_units * op->out_units, fp);
    fread(op->bias,    sizeof(float), op->out_units, fp);
}