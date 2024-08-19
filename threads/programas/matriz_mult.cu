/* Programa implementa el cálculo C = alpha * A * B + beta * C
 * que es parte de la libreria BLAS (SGEMM)
*/

#include <stdio.h>

#define BLOCK_DIM_X 4
#define BLOCK_DIM_Y 4

__global__ void
matriz_mult(const float *A, const float *B, float *C, int N, int M, int K, float alpha, float beta)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int fila = blockIdx.y * blockDim.y + threadIdx.y;

	float suma = 0.f;
	for (int i = 0; i < K; ++i) {
		suma += A[fila * K + i] * B[i * K + col];
	}
	
	C[fila * M + col] = alpha * suma + beta * C[fila * M + col];
}

void random_init(float *data, int size)
{
	for (int i = 0; i < size; ++i) {
		data[i] = (rand() & 0xFF) / (float)RAND_MAX;
	}
}

int main()
{
	float *A, *B, *C;
	float *d_A, *d_B, *d_C;
	int N, M, K;
	float alpha = 2.f;
	float beta = 1.f;
	N = M = K = 256;

    // asignación de memoria en el host
    A = (float *)malloc(N * K * sizeof(float));
    B = (float *)malloc(K * M * sizeof(float));
    C = (float *)malloc(N * M * sizeof(float));

	// asignación de memoria en el GPU
	cudaMalloc((void **)&d_A, N * K * sizeof(float));
	cudaMalloc((void **)&d_B, K * M * sizeof(float));
	cudaMalloc((void **)&d_C, N * M * sizeof(float));

	// incializar las matrices
	random_init(A, N * K);
	random_init(B, K * M);
	random_init(C, N * M);

	// copiar valores iniciales al GPU
	cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, A, K * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, A, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // invocar kernel
	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 dimGrid(M / dimBlock.x, N / dimBlock.y);
	matriz_mult << < dimGrid, dimBlock >> > (d_A, d_B, d_C, N, M, K, alpha, beta);

	// liberar espacio de memoria en el GPU
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// liberar memoria del host
	free(A);
	free(B);
	free(C);

	return 0;
}
