#include<stdio.h>

 // Kernel que suma los elementos de dos arrays
__global__
void suma(int n, float *x, float *y)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        y[i] = x[i] + y[i];
}
 
int main(void)
{
    int N = 1<<20;
    float *x, *y;

    // Asignar memoria unificada
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // inicializar los arrays x, y en el host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Invocar kernel. Notar que pasamos los arrays
    // que incializamos en el host - no hay 2 copias de los
    // arrays!
    int block = 256;
    int grid = (N + block - 1) / block;
    suma<<<grid, block>>>(N, x, y);

    // Esperar hasta que termine el GPU
    cudaDeviceSynchronize();

    // Verificar el resultado
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    printf("Max error: %f\n",maxError);

    // Liberar memoria
    cudaFree(x);
    cudaFree(y);

    return 0;
}
