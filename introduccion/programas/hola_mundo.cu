#include<stdio.h>
#include<stdlib.h> 

__global__ void imprimir_del_gpu() {
	printf("Hola Mundo! desde el thread [%d,%d] del device\n", threadIdx.x,blockIdx.x); 
}

int main() { 
	printf("Hola Mundo desde el host!\n"); 
	imprimir_del_gpu<<<1,10>>>();
	cudaDeviceSynchronize();
    return 0; 
}

