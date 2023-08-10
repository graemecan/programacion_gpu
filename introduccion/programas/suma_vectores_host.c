#include<stdio.h>
#include<stdlib.h>

#define N 512

void suma_host(int *a, int *b, int *c) {
	for(int idx=0;idx<N;idx++)
		c[idx] = a[idx] + b[idx];
}

// llenar el array con los indices
void llenar_array(int *data) {
	for(int idx=0;idx<N;idx++)
		data[idx] = idx;
}

void imprimir_salida(int *a, int *b, int*c) {
	for(int idx=0;idx<N;idx++)
		printf("\n %d + %d  = %d",  a[idx] , b[idx], c[idx]);
}
int main(void) {
	int *a, *b, *c;
	int size = N * sizeof(int);

	// Asignar memoria al lado del host para los arrays
	a = (int *)malloc(size); llenar_array(a);
	b = (int *)malloc(size); llenar_array(b);
	c = (int *)malloc(size);

	suma_host(a,b,c);

	imprimir_salida(a,b,c);

	free(a); free(b); free(c);


	return 0;
}
