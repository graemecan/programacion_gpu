#include <stdio.h>
#include <stdlib.h>

__global__ void
reduccion_memoria_global(float *data, int stride, int N)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_x + stride < N) {
        data[idx_x] += data[idx_x + stride];
    }
}

void inicializar_numeros(float *data, int size)
{
    for (int i = 0; i < size; i++)
    {
        // Generamos números aleatorios pequeños
        // El "0xFF" es un número en hexadecimal y está combinado con
        // rand() usando el "bitwise operator" & para obtener números
        // entre 0 y 255.
        data[i] = (float)(rand() & 0xFF) / (float)RAND_MAX;
    }
}

float resultado_cpu(float *data, int N)
{
    float resultado = 0.f;
    for (int i = 0; i < N; i++){
        resultado += data[i];
    }

    return resultado;
}

int main()
{
    float *h_array;
    float *d_array;

    unsigned int N = 1 << 25;
    unsigned int n_threads = 1024;

    float resultado_host, resultado_gpu;

    srand(2019);

    // Asignar memoria en el host
    h_array = (float *)malloc(N * sizeof(float));

    // Iniciliazar valores de h_array con números aleatorios
    inicializar_numeros(h_array, N);

    // Asignar memoria en el GPU y copiar datos
    cudaMalloc((void **)&d_array, N * sizeof(float));
    cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    // Calcular reducción en el GPU
    int n_bloques = (N + n_threads - 1) / n_threads;
    for (int stride = 1; stride < N; stride *= 2) {
        reduccion_memoria_global<<<n_bloques, n_threads>>>(d_array, stride, N);
    }

    // Copiar resultado del GPU
    cudaMemcpy(&resultado_gpu, &d_array[0], sizeof(float), cudaMemcpyDeviceToHost);

    // Calcular reducción en el CPU (secuencial)
    resultado_host = resultado_cpu(h_array, N);
    printf("host: %f, device %f\n", resultado_host, resultado_gpu);

    // Liberar memoria
    cudaFree(d_array);
    free(h_array);

    return 0;
}
