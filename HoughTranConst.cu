/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <cuda_runtime.h> // CUDA Events (Inciso 2)
#include "common/pgm.h"
#include <opencv2/opencv.hpp>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

void drawLinesOnImage(unsigned char *pixels, int* acc, int w, int h, float rMax, float rScale, float threshold)
{
    // Create a cv::Mat object from the input image data
    cv::Mat inputImage(h, w, CV_8U, pixels);

    // Create an output image with the same size and type as the input image
    cv::Mat output = inputImage.clone();

    // Draw lines on the output image
    for (int rIdx = 0; rIdx < rBins; ++rIdx)
    {
        for (int tIdx = 0; tIdx < degreeBins; ++tIdx)
        {
            int value = acc[rIdx * degreeBins + tIdx];

            // Check if the value is above the threshold
            if (value > threshold)
            {
                float r = rIdx * rScale - rMax;
                float theta = tIdx * radInc;

                // Calculate the coordinates of two points on the line
                int x1 = cvRound(w / 2 + r * cos(theta + M_PI / 2));
                int y1 = cvRound(h / 2 + r * sin(theta + M_PI / 2));
                int x2 = cvRound(w / 2 + r * cos(theta - M_PI / 2));
                int y2 = cvRound(h / 2 + r * sin(theta - M_PI / 2));

                // Draw the line on the output image
                cv::line(output, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 1);
            }
        }
    }

    // Save the output image
    cv::imwrite("output.jpg", output);
}


//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
    *acc = new int[rBins * degreeBins];                // el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
    memset(*acc, 0, sizeof(int) * rBins * degreeBins); // init en ceros
    int xCent = w / 2;
    int yCent = h / 2;
    float rScale = 2 * rMax / rBins;

    for (int i = 0; i < w; i++)     // por cada pixel
        for (int j = 0; j < h; j++) //...
        {
            int idx = j * w + i;
            if (pic[idx] > 0) // si pasa thresh, entonces lo marca
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;                       // y-coord has to be reversed
                float theta = 0;                              // actual angle
                for (int tIdx = 0; tIdx < degreeBins; tIdx++) // add 1 to all lines in that pixel
                {
                    float r = xCoord * cos(theta) + yCoord * sin(theta);
                    int rIdx = (r + rMax) / rScale;
                    (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                    theta += radInc;
                }
            }
        }
}

//*****************************************************************
// usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];

//*****************************************************************
// TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }

//  Kernel memoria Constante
__global__ void GPU_HoughTranConst(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
   // Calculo gloID (Inciso 1)
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h)
        return;

    int xCent = w / 2;
    int yCent = h / 2;

    //  Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // eventualmente usar memoria compartida para el acumulador

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            // utilizar memoria constante para senos y cosenos
             float r = xCoord * cosf(tIdx) + yCoord * sinf(tIdx); //probar con esto para ver diferencia en tiempo
            //float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
    // Calculo gloID (Inciso 1)
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    if (gloID >= w * h)
        return;

    int xCent = w / 2;
    int yCent = h / 2;

    // Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // TODO eventualmente usar memoria compartida para el acumulador

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            // utilizar memoria constante para senos y cosenos
            //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
            float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (r + rMax) / rScale;
            // debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
            atomicAdd(acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

    // TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
    // utilizar operaciones atomicas para seguridad
    // faltara sincronizar los hilos del bloque en algunos lados
}

//*****************************************************************
int main(int argc, char **argv)
{
    int i;

    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    float *d_Cos;
    float *d_Sin;

    //cudaMalloc((void **)&d_Cos, sizeof(float) * degreeBins);
    //cudaMalloc((void **)&d_Sin, sizeof(float) * degreeBins);

    // CPU calculation
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // pre-compute values to be stored
    float *pcCos = (float *)malloc(sizeof(float) * degreeBins);
    float *pcSin = (float *)malloc(sizeof(float) * degreeBins);
    float rad = 0;

    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(float) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(float) * degreeBins);
    
    for (i = 0; i < degreeBins; i++)
    {
        pcCos[i] = cos(rad);
        pcSin[i] = sin(rad);
        rad += radInc;
    }

    float rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2;
    float rScale = 2 * rMax / rBins;

    // TODO eventualmente volver memoria global
    //cudaMemcpy(d_Cos, pcCos, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Sin, pcSin, sizeof(float) * degreeBins, cudaMemcpyHostToDevice);

    // setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // CUDA Events (Inciso 2)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CUDA Event empezar record
    cudaEventRecord(start, 0);

    // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
    // 1 thread por pixel
    int blockNum = ceil(w * h / 256);
    //GPU_HoughTran<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
    GPU_HoughTranConst<<<blockNum, 256>>>(d_in, w, h, d_hough, rMax, rScale);

    // CUDA Event terminar record
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);


    // get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // compare CPU and GPU results
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }

    // Inciso 4 - Generacion de imagen output.jpg
    float threshold = 5;
    drawLinesOnImage(inImg.pixels, h_hough, w, h, rMax, rScale, threshold);
    printf("Done!\n");
    printf("Time taken by GPU_HoughTranConst: %f ms\n", elapsedTime);


    // Liberar memoria en el host
    free(cpuht);
    free(pcCos);
    free(pcSin);
    free(h_hough);

    // Liberar memoria en el dispositivo (Inciso 1)
    //cudaFree(d_Cos);
    //cudaFree(d_Sin);
    cudaFree(d_in);
    cudaFree(d_hough);

    // Liberar eventos
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
  }