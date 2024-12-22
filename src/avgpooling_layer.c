#include "avgpooling_layer.h"

void avg_pooling_forward(avg_pooling *op) {    
    for (int c=0;c<op->channels;c++){
    for (int i = 0; i < op->out_h; ++i) {
        for (int j = 0; j < op->out_w; ++j) {
            float sum = 0.0;
            for (int k = 0; k < op->kernel_size; ++k) {
                for (int l = 0; l < op->kernel_size; ++l) {
                    int input_row = i * op->stride + k;
                    int input_col = j * op->stride + l;
                    sum += op->input[input_row * op->in_w + input_col];
                }
            }
            op->output[c*op->out_h*op->out_w+i * op->out_w + j] = sum / (op->kernel_size * op->kernel_size);
        }
    }
    }
}