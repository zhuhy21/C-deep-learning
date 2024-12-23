# C-deep-learning  

I implemented deep learning network inference using C programming language.(including AlexNet, ResNet & VGGNet)  
## HOW TO USE  
You need to write the correct path for weights and dataset in the main() of alexnet.c/resnet.c/vggnet.c.
The weights must be in a correct format which is wrote in the load() function. You can change a .pth into a .weights manually.  

use alexnet:

    make alexnet
    make inf_alexnet
use resnet:

    make resnet
    make inf_resnet
use vggnet:

    make vggnet
    make inf_vggnet
## DETAILS  
I realized convolution layer, activation layer, pooling layer and fully connected layer in this program.  
You can build your own network by using this module.  
## EVAL  
TO BE PUBLISHED
