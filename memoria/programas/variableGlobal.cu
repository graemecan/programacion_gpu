#include <stdio.h>

#define N 10

__device__ int devVar[N];

__global__ void modificarVariableGlobal()
{
    // modificar los valores
    devVar[threadIdx.x] += 2;
}

int main(void)
{
    // inicializar variable global
    int hostVar[N];
    for (int i = 0; i < N; i++){
        hostVar[i] = i;
    }

    // copiar valores al device (hay que usar "cudaMemcpyToSymbol")
    cudaMemcpyToSymbol(devVar, &hostVar, N*sizeof(int));
    for (int i = 0; i < N; i++){
        printf("Antes del kernel: %d\n", hostVar[i]);
    }

    // invocar el kernel
    modificarVariableGlobal<<<1, N>>>();

    // copiar valores del device al host
    cudaMemcpyFromSymbol(&hostVar, devVar, N*sizeof(int));
    for (int i = 0; i < N; i++){
        printf("Después del kernel: %d\n", hostVar[i]);
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
