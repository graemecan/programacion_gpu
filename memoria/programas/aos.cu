#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<string.h>


#define NUM_THREADS 256

#define IMG_SIZE 1048576

// Array of Structures
struct Datos_AOS {
  int r;
  int b;
  int g;
  int hue;
  int saturation;
  int maxVal;
  int minVal;
  int finalVal; 
};


__global__
void calculoComplicado(Datos_AOS*  datos)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;


  int grayscale = (datos[i].r + datos[i].g + datos[i].b)/datos[i].maxVal;
  int hue_sat = datos[i].hue * datos[i].saturation / datos[i].minVal;
  datos[i].finalVal = grayscale*hue_sat; 
}

void calculoComplicado()
{

  Datos_AOS* d_x;

  cudaMalloc(&d_x, IMG_SIZE*sizeof(Datos_AOS)); 

  int num_blocks = IMG_SIZE/NUM_THREADS;

  calculoComplicado<<<num_blocks,NUM_THREADS>>>(d_x);

  cudaFree(d_x);
}



int main()
{

	calculoComplicado();
	return 0;
}






