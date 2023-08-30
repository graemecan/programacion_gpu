#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BDIM 32

void inicializarMatriz(float *entrada,  const int N)
{
    for (int i = 0; i < N; i++)
    {
        entrada[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

void verificarResultado(float *hostRef, float *gpuRef, const int Ntot)
{
    double epsilon = 1.0E-8;
    bool igual = 1;

    for (int i = 0; i < Ntot; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            igual = 0;
            printf("diferencia en elemento %d: host %f gpu %f\n", i, hostRef[i],
                   gpuRef[i]);
            break;
        }
    }

    if (!igual)  printf("Matrices no coinciden.\n\n");
}

void transpuestaHost(float *salida, float *entrada, const int N)
{
    for( int iy = 0; iy < N; ++iy)
    {
        for( int ix = 0; ix < N; ++ix)
        {
            salida[ix * N + iy] = entrada[iy * N + ix];
        }
    }
}

// Transpuesta con memoria global (cargar por columnas)
__global__ void transpuestaGlobal(float *salida, float *entrada, const int N)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < N && iy < N)
    {
        salida[iy * N + ix] = entrada[ix * N + iy];
    }
}

__global__ void transpuestaComp(float *salida, float *entrada, int N)
{
    // Memoria compartida estática.
    // La palabra "tile" se refiere a una cerámica 
    // del baño/cocina, cómo un cuadrito.
    __shared__ float tile[BDIM][BDIM];

    // coordenadas globales en la matriz original
    unsigned int ix, iy, ti, to, ixt, iyt;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // coordenada en memoria lineal
    ti = iy * N + ix;

    // coordenada en matriz transpuesta
    // Aquí tenemos "transpuesta de bloques"
    ixt = blockDim.y * blockIdx.y + threadIdx.x;
    iyt = blockDim.x * blockIdx.x + threadIdx.y;

    // coordenada global lineal en matriz transpuesta
    to = iyt * N + ixt;

    // determinar transpuesta
    if (ixt < N && iyt < N)
    {
        // cargar datos de memoria global a memoria compartida
        tile[threadIdx.y][threadIdx.x] = entrada[ti];

        // sincronización de los threads en el bloque
        __syncthreads();

        // guardar datos a memoria global de memoria compartida
        salida[to] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpuestaCompDin(float *salida, float *entrada, int N)
{
    // memoria compartida dinámica
    extern __shared__ float tile[];

    // coordenadas de la matriz original
    unsigned int  ix, iy, ti, to;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // coordenada lineal de la matriz original
    ti = iy * N + ix;

    // indice del thread en bloque transpuesto
    unsigned int fila_idx, col_idx;
    fila_idx = threadIdx.y * blockDim.x + threadIdx.x;
    col_idx = threadIdx.x * blockDim.x + threadIdx.y;

    // coordenadas en la matriz transpuesta
    ix = blockDim.y * blockIdx.y + threadIdx.x;
    iy = blockDim.x * blockIdx.x + threadIdx.y;

    // coordenada lineal de la matriz transpuesta
    to = iy * N + ix;

    // calcular transpuesta
    if (ix < N && iy < N)
    {
        // cargar de memory global a memoria compartida
        tile[fila_idx] = entrada[ti];

        // sincronizar el bloque
        __syncthreads();

        // guardar a memoria global de memoria compartida
        salida[to] = tile[col_idx];
    }
}

__global__ void transpuestaCompPad(float *salida, float *entrada, int N)
{
    // Memoria compartida estática.
    // La palabra "tile" se refiere a una cerámica 
    // del baño/cocina, cómo un cuadrito.
    __shared__ float tile[BDIM][BDIM+1];

    // coordenadas globales en la matriz original
    unsigned int ix, iy, ti, to, ixt, iyt;
    ix = blockDim.x * blockIdx.x + threadIdx.x;
    iy = blockDim.y * blockIdx.y + threadIdx.y;

    // coordenada en memoria lineal
    ti = iy * N + ix;

    // coordenada en matriz transpuesta
    // Aquí tenemos "transpuesta de bloques"
    ixt = blockDim.y * blockIdx.y + threadIdx.x;
    iyt = blockDim.x * blockIdx.x + threadIdx.y;

    // coordenada global lineal en matriz transpuesta
    to = iyt * N + ixt;

    // determinar transpuesta
    if (ixt < N && iyt < N)
    {
        // cargar datos de memoria global a memoria compartida
        tile[threadIdx.y][threadIdx.x] = entrada[ti];

        // sincronización de los threads en el bloque
        __syncthreads();

        // guardar datos a memoria global de memoria compartida
        salida[to] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char **argv)
{

    // matriz de 4096x4096
    int N = 1 << 12;

    printf("Dimensiones de matrices: nx %d ny %d\n", N, N);
    int nBytes = N * N * sizeof(float);

    // configuración del kernel
    dim3 block (BDIM, BDIM);
    dim3 grid  ((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // asignar memoria en el host
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    //  inicializar matriz en el host
    inicializarMatriz(h_A, N * N);

    //  transpuesta calculada en el host
    double comienzo = segundos();
    transpuestaHost(hostRef, h_A, N);
    double tiempo_total = segundos() - comienzo;
    printf("transpuestaHost tiempo %f s.\n", tiempo_total);

    // asignar memoria en el device
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copiar datos del host al device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // invocar kernel que usa memoria global
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    comienzo = segundos();
    transpuestaGlobal<<<grid, block>>>(d_C, d_A, N);
    CHECK(cudaDeviceSynchronize());
    tiempo_total = segundos() - comienzo;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    verificarResultado(hostRef, gpuRef, N);
    float ibnd = 2 * N * N * sizeof(float) / 1e9 / tiempo_total;
    printf("transpuestaGlobal tiempo %f s <<< grid (%d,%d) block (%d,%d)>>> "
           "bandwidth efectivo %f GB/s\n", tiempo_total, grid.x, grid.y, block.x,
           block.y, ibnd);

    // invocar kernel que usa memoria compartida
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    comienzo = segundos();
    transpuestaComp<<<grid, block>>>(d_C, d_A, N);
    CHECK(cudaDeviceSynchronize());
    tiempo_total = segundos() - comienzo;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    verificarResultado(hostRef, gpuRef, N);
    ibnd = 2 * N * N * sizeof(float) / 1e9 / tiempo_total;
    printf("transpuestaComp tiempo %f s <<< grid (%d,%d) block (%d,%d)>>> "
           "bandwidth efectivo %f GB/s\n", tiempo_total, grid.x, grid.y, block.x,
           block.y, ibnd);

    // invocar kernel que usa memoria compartida dinámica
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    comienzo = segundos();
    transpuestaCompDin<<<grid, block, BDIM*BDIM*sizeof(float)>>>(d_C, d_A, N);
    CHECK(cudaDeviceSynchronize());
    tiempo_total = segundos() - comienzo;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    verificarResultado(hostRef, gpuRef, N);
    ibnd = 2 * N * N * sizeof(float) / 1e9 / tiempo_total;
    printf("transpuestaCompDin tiempo %f s <<< grid (%d,%d) block (%d,%d)>>> "
           "bandwidth efectivo %f GB/s\n", tiempo_total, grid.x, grid.y, block.x,
           block.y, ibnd);

    // invocar kernel que usa memoria compartida dinámica
    CHECK(cudaMemset(d_C, 0, nBytes));
    memset(gpuRef, 0, nBytes);

    comienzo = segundos();
    transpuestaCompPad<<<grid, block, BDIM*(BDIM+1)*sizeof(float)>>>(d_C, d_A, N);
    CHECK(cudaDeviceSynchronize());
    tiempo_total = segundos() - comienzo;

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    verificarResultado(hostRef, gpuRef, N);
    ibnd = 2 * N * N * sizeof(float) / 1e9 / tiempo_total;
    printf("transpuestaCompPad tiempo %f s <<< grid (%d,%d) block (%d,%d)>>> "
           "bandwidth efectivo %f GB/s\n", tiempo_total, grid.x, grid.y, block.x,
           block.y, ibnd);

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
