#include "common.h"
#include <stdio.h>

// función para inicializar las matrices con números aleatorios
void inicializarMatriz(float *matriz,  const int N)
{
    for (int i = 0; i < N; i++)
    {
        matriz[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

// kernel para copiar los valores de una matriz a otra, fila por fila
__global__ void copiarFila(float *salida, float *entrada, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        salida[iy * nx + ix] = entrada[iy * nx + ix];
    }
}

int main()
{

    // inicializar matrices con 2048x2048 elementos
    int nx = 1 << 11;
    int ny = 1 << 11;

    // tamaños de los bloques 16x16
    int blockx = 16;
    int blocky = 16;

    // número de bytes de datos en las matrices
    int nBytes = nx * ny * sizeof(float);

    // determinar configuración para el kernel
    dim3 block (blockx, blocky);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // asignar memoria del host
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // inicializar memoria del host
    inicializarMatriz(h_A, nx * ny);

    // asignar memoria del device
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copiar datos del host al device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // ejecutar el kernel una vez para inicializar el uso del GPU
    double comienzo = segundos();
    copiarFila<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double tiempo_total = segundos() - comienzo;
    printf("Tiempo para correr kernel primera vez: %f s\n", tiempo_total);
    CHECK(cudaGetLastError());

    // ejecutar el kernel de nuevo
    comienzo = segundos();
    copiarFila<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    tiempo_total = segundos() - comienzo;

    // calcular bandwidth efectivo
    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / tiempo_total;
    printf("copiarFila demoró  %f s <<< grid (%d,%d) block (%d,%d)>>> bandwidth "
           "efectivo %f GB/s\n", tiempo_total, grid.x, grid.y, block.x,
           block.y, ibnd);
    CHECK(cudaGetLastError());

    // liberar memoria del host y del device
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
