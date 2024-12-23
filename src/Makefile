alexnet:
	gcc -o alexnet alexnet.c data.c convolution_layer.c activation_layer.c maxpooling_layer.c fc_layer.c

vggnet:
	gcc -o vggnet resnet.c data.c convolution_layer.c activation_layer.c maxpooling_layer.c fc_layer.c

resnet:
	gcc -o resnet resnet.c data.c convolution_layer.c activation_layer.c maxpooling_layer.c fc_layer.c avgpooling_layer.c

inf_alexnet:
	./alexnet

inf_vggnet:
	./vggnet
	
inf_resnet:
	./resnet

clean: 
	rm alexnet
	rm vggnet
	rm resnet