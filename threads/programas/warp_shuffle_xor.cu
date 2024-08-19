#include <stdio.h>
#define FULL_MASK 0xffffffff

__global__ void intercambio_gpu(int *data)
{
    int val = data[threadIdx.x];
    // Intercambiar con el "lane" que tiene ID que resulta
    // de un XOR combinando FULL_MASK con el "lane" actual.
    val = __shfl_xor_sync(FULL_MASK, val, FULL_MASK);
    data[threadIdx.x] = val;
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

	int *h_data, *d_data;

    // asignación de memoria en el host
    h_data = (int *)malloc(N * sizeof(int));

	// asignación de memoria en el GPU
	cudaMalloc((void **)&d_data, N * sizeof(int));

	// incializar el array
	inicializar(h_data, N);

    for (int i = 0; i < N; i++){    
        printf("CPU: %d %d\n",i,h_data[i]);
    }

	// copiar valores iniciales al GPU
	cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // invocar kernel
	intercambio_gpu<<<n_bloques, n_threads>>>(d_data);

    // copiar resultado del GPU al host
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++){    
        printf("GPU: %d %d\n",i,h_data[i]);
    }

	// liberar espacio de memoria en el GPU
	cudaFree(d_data);

	// liberar memoria del host
	free(h_data);

	return 0;
}
