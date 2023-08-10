#include<stdio.h>
#include<stdlib.h>

#define N 2048

void suma_host(int *a, int *b, int *c) {
	for(int idx=0;idx<N;idx++)
		c[idx] = a[idx] + b[idx];
}

__global__ void suma_device(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}


// llenar el array con los indices
void llenar_array(int *data) {
	for(int idx=0;idx<N;idx++)
		data[idx] = idx;
}

void imprimir_salida(int *a, int *b, int*c) {
	for(int idx=0;idx<N;idx++)
		printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}
int main(void) {
	int *a, *b, *c;
    int *d_a, *d_b, *d_c; // copias de los arrays en el device

	int size = N * sizeof(int);

	// Asignar memoria al lado del host
	a = (int *)malloc(size); llenar_array(a);
	b = (int *)malloc(size); llenar_array(b);
	c = (int *)malloc(size);

    // Asignar memoria al lado del device (GPU)
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copiar al device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Invocar kernel con N bloques de 1 thread cada uno
	suma_device<<<2,N/2>>>(d_a,d_b,d_c);
    cudaError_t err = cudaGetLastError();
    printf("Error: %s\n",cudaGetErrorString(err));

    // Copiar resultado al host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	imprimir_salida(a,b,c);

	free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}
