#gcc -o data.o -c data.c
#gcc data.o -o data
gcc -o main alexnet.c data.c convolution_layer.c activation_layer.c maxpooling_layer.c fc_layer.c