#include <stdio.h>

#define N 10

__global__ void modificarVariableGlobal(int* devVar)
{
    // modificar los valores
    devVar[threadIdx.x] += 2;
}

int main(void)
{
    // inicializar variable global
    int* hostVar = (int *) malloc(N*sizeof(int));
    int* devVar;
    cudaMalloc((int**)&devVar, N*sizeof(int));

    for (int i = 0; i < N; i++){
        hostVar[i] = i;
    }

    // copiar valores al device (hay que usar "cudaMemcpyToSymbol")
    cudaMemcpy(devVar, hostVar, N*sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < N; i++){
        printf("Antes del kernel: %d\n", hostVar[i]);
    }

    // invocar el kernel
    modificarVariableGlobal<<<1, N>>>(devVar);

    // copiar valores del device al host
    cudaMemcpy(hostVar, devVar, N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++){
        printf("Después del kernel: %d\n", hostVar[i]);
    }

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
