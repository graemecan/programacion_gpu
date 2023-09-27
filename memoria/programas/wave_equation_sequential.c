#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>

double gaussian(double x, double t, double c){

    double mean = 200.0;
    double sig = 10.0;
    double normalise = 1.0;

    return (1.0/normalise)*exp(-0.5*(((x-c*t) - mean)*((x-c*t) - mean))/(sig*sig));

}

void initialise_arrays(double* un, double* un_m1, double* un_p1, int N, double delta_x, double delta_t, double t, double c){

    for (int i = 0; i < N; i++){

        double xpos = i*delta_x;

        un[i] = gaussian(xpos, t, c);
        un_p1[i] = 0.0;
        un_m1[i] = gaussian(xpos, t-delta_t, c);

    }

}

void main(){

    double c, delta_t, delta_x, t_ini;

    c = 1.0;
    delta_t = 0.1;
    delta_x = 1.0;

    t_ini = 0.0;

    double cfl = c*c*delta_t*delta_t/(delta_x*delta_x);

    int Ni = 10000;
    int Nt = 20000;
    int array_size = sizeof(double)*Ni;

    int count = 0;

    double *un = malloc(array_size);
    double *un_p1 = malloc(array_size);
    double *un_m1 = malloc(array_size);
    double *tmp;

    FILE* datafile;
    char filename[20];

    initialise_arrays(un, un_m1, un_p1, Ni, delta_x, delta_t, t_ini, c);

    for (int t = 0; t < Nt; t++){

        if (t%1000 == 0) {
            sprintf(filename, "u_%05d.dat", count);
            count += 1;

            datafile = fopen(filename,"w");
            fwrite(&(un[0]),sizeof(double),Ni,datafile);
            fclose(datafile);
        }

        for (int i = 0; i < Ni; i++){

            un_p1[i] = cfl*(un[i+1] + un[i-1] - 2.0*un[i]) - un_m1[i] + 2.0*un[i];

        }

        //memcpy(un_m1, un, array_size);
        //memcpy(un, un_p1, array_size);
        tmp = un;
        un_m1 = un;
        un = un_p1;
        un_p1 = tmp;

    }

}
