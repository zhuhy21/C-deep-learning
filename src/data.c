#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "data.h"

long get_filesize(FILE* fp)
{
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    return filesize;
}

typedef enum Endian {
    ENDIAN_LSB = 0,
    ENDIAN_MSB = 1
} Endian;

int read_int_from_4_bytes(unsigned char* buf, Endian endian)
{
    int x = 0;
    int c[2][4] = {
        { (1 << 0),  (1 << 8), (1 << 16), (1 << 24) },
        { (1 << 24), (1 << 16), (1 << 8), (1 << 0) }
    };
    for (int i=0; i<4; i++)
        x += buf[i] * c[endian][i];
    return x;
}



void destroy_dataset(DataSet* dataset)
{
    if (dataset)
    {
        free(dataset->image_buf);
        dataset->image_buf = NULL;

        free(dataset->label_buf);
        dataset->labels = NULL;
        
        free(dataset->images);
        dataset->images = NULL;
    }
}

void load_labels(DataSet* dataset, const char* filename)
{
    FILE* fin = fopen(filename, "rb");
    long filesize = get_filesize(fin);
    unsigned char* buf = (unsigned char*)malloc(filesize + 1);
    if (buf == NULL)
        exit(1);
    buf[filesize] = '\0';
    dataset->label_buf = buf;
    fread((void*)buf, filesize, 1, fin);
    fclose(fin);
    dataset->num_labels = read_int_from_4_bytes(buf + 4, ENDIAN_MSB);
    dataset->labels = buf + 8;
}

void load_images(DataSet* dataset, const char* filename)
{
    
    FILE* fin = fopen(filename, "rb");
    if(fin==NULL){
        printf("NO!\n");
    }
    long filesize = get_filesize(fin);
    unsigned char* buf = (unsigned char*)malloc(filesize + 1);
    if (buf == NULL){
        printf("error\n");
        exit(1);
    }
    dataset->image_buf = buf;
    
    buf[filesize] = '\0';
    fread((void*)buf, filesize, 1, fin);
    fclose(fin);
    
    uint8_t magic[4] = { buf[0], buf[1], buf[2], buf[3] };

    int num_images = read_int_from_4_bytes(buf + 4, ENDIAN_MSB);
    int rows = read_int_from_4_bytes(buf + 8, ENDIAN_MSB);
    int cols = read_int_from_4_bytes(buf + 12, ENDIAN_MSB);
    
    DE_GrayImage* images = (DE_GrayImage*)malloc(sizeof(DE_GrayImage) * num_images);
    if (images == NULL) 
        exit(1);
    dataset->images = images;
    for (int i=0; i<num_images; i++)
    {
        images[i].height = rows;
        images[i].width = cols;
        images[i].data = buf + 16 + i * rows * cols;
    }
}
//DEBUG ONLY


// void print_sample(const DataSet* dataset, int index)
// {
//     DE_GrayImage* image = &dataset->images[index];

//     printf("label: %d\n", (int)dataset->labels[index]);
//     for (int i=0; i<28; i++)
//     {
//         for (int j=0; j<28; j++)
//         {
//             for (int k=0; k<3;k++)
//                 printf("%c", image->data[i * 28 + j] > 128 ? '#':' ');
//         }
//         printf("\n");
//     }
// }

// int main()
// {
//     DataSet dataset;
//     load_images(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/train-images.idx3-ubyte");
//     load_labels(&dataset, "/share/public/zhuhongyu/Unixhw/MINST/train-labels.idx1-ubyte");
    
//     print_sample(&dataset, 0);
//     print_sample(&dataset, 233);
//     print_sample(&dataset, 666);

//     printf("wait\n");
//     destroy_dataset(&dataset);

//     return 0;
// }
