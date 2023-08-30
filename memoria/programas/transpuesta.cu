#include "common.h"
#include <stdio.h>

void inicializarMatriz(float *entrada,  const int N)
{
    for (int i = 0; i < N; i++)
    {
        entrada[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void verificarResultado(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool igual = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            igual = 0;
            printf("diferencia en el elemento %d: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }
    }

    if (!igual)  printf("Resultados no coinciden.\n\n");
}

void transpuestaHost(float *salida, float *entrada, const int nx, const int ny)
{
    for( int iy = 0; iy < ny; ++iy)
    {
        for( int ix = 0; ix < nx; ++ix)
        {
            salida[ix * ny + iy] = entrada[iy * nx + ix];
        }
    }
}

// Guardar por columnas, cargar por filas
__global__ void transpuestaCargarFilas(float *salida, float *entrada, const int nx,
                                  const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        salida[ix * ny + iy] = entrada[iy * nx + ix];
    }
}

// Guardar por filas, cargar por columnas
__global__ void transpuestaCargarColumnas(float *salida, float *entrada, const int nx,
                                  const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        salida[iy * nx + ix] = entrada[ix * ny + iy];
    }
}

// main functions
int main()
{

    // dimensiones de las matrices: 2048x2048
    int nx = 1 << 11;
    int ny = 1 << 11;

    // tamaño de bloque
    int blockx = 16;
    int blocky = 16;

    // número de bytes en las matrices
    int nBytes = nx * ny * sizeof(float);

    // configuración del kernel
    dim3 block (blockx, blocky);
    dim3 grid  ((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // asignar memoria del host
    float *h_A = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // poner números en la matriz
    inicializarMatriz(h_A, nx * ny);

    // transpuesta al lado del host
    transpuestaHost(hostRef, h_A, nx, ny);

    // asignar memoria del device
    float *d_A, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copiar datos del host al device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // invocar un kernel por primera vez
    double comienzo = segundos();
    transpuestaCargarFilas<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double tiempo_total = segundos() - comienzo;
    printf("Tiempo del primer kernel: %f s\n", tiempo_total);
    CHECK(cudaGetLastError());

    // invocar kernel de cargar por filas
    comienzo = segundos();
    transpuestaCargarFilas<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    tiempo_total = segundos() - comienzo;

    // calcular bandwidth efectivo
    float ibnd = 2 * nx * ny * sizeof(float) / 1e9 / tiempo_total;
    printf("transpuestaCargarFilas: %f s <<< grid (%d,%d) block (%d,%d)>>> bandwidth "
           "efectivo %f GB\n", tiempo_total, grid.x, grid.y, block.x,
           block.y, ibnd);
    CHECK(cudaGetLastError());

    // verificar resultados
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    verificarResultado(hostRef, gpuRef, nx * ny);

    // copiar datos del host al device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    // invocar kernel de cargar por columnas
    comienzo = segundos();
    transpuestaCargarColumnas<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaDeviceSynchronize());
    tiempo_total = segundos() - comienzo;

    // calcular bandwidth efectivo
    ibnd = 2 * nx * ny * sizeof(float) / 1e9 / tiempo_total;
    printf("transpuestaCargarColumnas: %f s <<< grid (%d,%d) block (%d,%d)>>> bandwidth "
           "efectivo %f GB\n", tiempo_total, grid.x, grid.y, block.x,
           block.y, ibnd);
    CHECK(cudaGetLastError());

    // verificar resultados
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    verificarResultado(hostRef, gpuRef, nx * ny);

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
