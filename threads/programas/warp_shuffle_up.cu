#include <stdio.h>
#define FULL_MASK 0xffffffff

__global__ void reduccion_gpu(int *data, int N)
{
    int val = data[threadIdx.x];
	for (int stride = 16; stride > 0; stride >>= 1){
        val -= __shfl_up_sync(FULL_MASK, val, stride);
    }
    if (threadIdx.x == warpSize-1) data[0] = val;
}

int reduccion_cpu(int *data, int N)
{
    int resultado = 0;
    for (int i = N-1; i >= 0; i--){
        resultado -= data[i];
    }
    return resultado;
}

void inicializar(int *data, int N)
{
	for (int i = 0; i < N; ++i) {
		data[i] = i;
	}
}

int main()
{
    // Este programa no funciona para N != 32...
    int N = 1 << 5;

    // Usamos un "warp" de threads solamente
    int n_threads = 32;
    int n_bloques = 1;

    int resultado_cpu, resultado_gpu;
	int *h_data, *d_data;

    // asignación de memoria en el host
    h_data = (int *)malloc(N * sizeof(int));

	// asignación de memoria en el GPU
	cudaMalloc((void **)&d_data, N * sizeof(int));

	// incializar el array
	inicializar(h_data, N);

	// copiar valores iniciales al GPU
	cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // invocar kernel
	reduccion_gpu<<<n_bloques, n_threads>>>(d_data, N);

    // copiar resultado del GPU al host
    cudaMemcpy(&resultado_gpu, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // calcular en el host para comparar
    resultado_cpu = reduccion_cpu(h_data, N);
    
    printf("Host: %d, GPU: %d\n",resultado_cpu,resultado_gpu);

	// liberar espacio de memoria en el GPU
	cudaFree(d_data);

	// liberar memoria del host
	free(h_data);

	return 0;
}
