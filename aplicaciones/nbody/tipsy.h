#ifndef __TIPSY_H__
#define __TIPSY_H__

#include <stdio.h>
#define MAXDIM 3

typedef float Real;

struct gas_particle
{
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real rho;
    Real temp;
    Real hsmooth;
    Real metals ;
    Real phi ;
} ;

//struct gas_particle *gas_particles;

struct dark_particle
{
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real eps;
    int phi ;
} ;

//struct dark_particle *dark_particles;

struct star_particle
{
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real metals ;
    Real tform ;
    Real eps;
    int phi ;
} ;

//struct star_particle *star_particles;

struct dump
{
    double time ;
    int nbodies ;
    int ndim ;
    int nsph ;
    int ndark ;
    int nstar ;
} ;

typedef struct dump header ;

void read_tipsy_file(float4* bodyPosition,
                     float4* bodyVelocity,
                     char* fileName, int numBodies)
{
    /*
       Read in our custom version of the tipsy file format written by
       Jeroen Bedorf.  Most important change is that we store particle id on the
       location where previously the potential was stored.
    */


    // printf("Trying to read file: %s\n",fileName);

    FILE *fptr;

    if ((fptr = fopen(fileName,"rb")) == NULL)
    {
        printf("Can't open input file \n");
        exit(EXIT_SUCCESS);
    }

    struct dump h;
    fread((char *)&h, sizeof(h), 1, fptr);

    // int idummy;

    //Read tipsy header
    int NTotal    = h.nbodies;
    int NFirst    = h.ndark;
    int NSecond   = h.nstar;
    int NThird    = h.nsph;

    printf("%d %d %d %d\n",NTotal,NFirst,NSecond,NThird);

    // round up to a multiple of 256 bodies since our kernel only supports that...
    int newTotal = NTotal;
    // int newTotal = NSecond;

    if (newTotal % 256)
    {
        newTotal = ((newTotal / 256) + 1) * 256;
    }

    if (newTotal != numBodies){
      printf("Stated number of particles in global numBodies variable does not match file+padding.\n");
      printf("numBodies = %d, newTotal = %d\n",numBodies,newTotal);
      exit(99);
    }

    //Start reading
    int particleCount = 0;

    struct dark_particle d;
    struct star_particle s;

    for (int i=0; i < NFirst+NSecond; i++)
    {
        if (i < NFirst)
        {
            fread((char *)&d, sizeof(d), 1, fptr);
            bodyVelocity[i].w        = d.eps;
            bodyPosition[i].w       = d.mass;
            bodyPosition[i].x       = d.pos[0];
            bodyPosition[i].y       = d.pos[1];
            bodyPosition[i].z       = d.pos[2];
            bodyVelocity[i].x        = d.vel[0];
            bodyVelocity[i].y        = d.vel[1];
            bodyVelocity[i].z        = d.vel[2];
            // idummy            = d.phi;
        }
        else
        {
            fread((char *)&s, sizeof(s), 1, fptr);
            bodyVelocity[i].w        = s.eps;
            bodyPosition[i].w       = s.mass;
            bodyPosition[i].x       = s.pos[0];
            bodyPosition[i].y       = s.pos[1];
            bodyPosition[i].z       = s.pos[2];
            bodyVelocity[i].x        = s.vel[0];
            bodyVelocity[i].y        = s.vel[1];
            bodyVelocity[i].z        = s.vel[2];
            // idummy            = s.phi;
        }

        // bodiesIDs[i] = idummy;
        particleCount++;
    }//end for

    for (int i = NSecond; i < newTotal; i++)
    {
        bodyPosition[i].w = 0.0;
        bodyPosition[i].x = bodyPosition[i].y = bodyPosition[i].z = 0.0;
        bodyVelocity[i].x = bodyVelocity[i].y = bodyVelocity[i].z = 0.0;
        // bodiesIDs[i] = i;
        // NFirst++;
        particleCount++;
    }

    fclose(fptr);
}

#endif //__TIPSY_H__
