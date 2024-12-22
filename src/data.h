#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef struct DE_GrayImage
{
    unsigned int width;
    unsigned int height;
    unsigned char* data;
} DE_GrayImage;

typedef struct DataSet
{
    DE_GrayImage* images;
    uint8_t* labels;
    uint8_t* image_buf;
    uint8_t* label_buf;
    int num_images;
    int num_labels;
} DataSet;

void load_labels(DataSet* dataset, const char* filename);
void load_images(DataSet* dataset, const char* filename);