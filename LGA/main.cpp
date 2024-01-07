#include<iostream>
#include<GL\glew.h>
#include<glfw3.h>
#include".\inc\LGA_Simulation.h"
int main()
{
    srand(time(NULL));
   
    LBM_Config* config = (LBM_Config*)malloc(sizeof(LBM_Config));
    config->domain_Device = NULL;
    config->domain_Host = NULL;
    config->isWorking = 0;
    config->shutDown = 0;
    config->nx = 64;
    config->ny = 64;

    config->flow = 1;
    config->dynamicViscousity = 1.867e-3;
    config->fluidRo = 1.286;
    config->dx = 0.1 / (double)config->nx;
    config->dt = config->dx * 0.1;
    config->defaultRo = 0.2; 1e-12;
    config->field = 0;

    config->visualisation.field = 0;
    config->visualisation.amplifier = 1.0;
    config->visualisation.amplifierStep = 0.1;

    config->boundryN.type = 1;
    config->boundryN.normalVelocity = 0.2;

    config->boundryS.type = 1;
    config->boundryS.normalVelocity = 0.01;
    
    config->boundryE.type = 4;
    config->boundryE.normalVelocity = -0.01;

    config->boundryW.type = 4;
    config->boundryW.normalVelocity = 0.01;

    cudaMalloc(&config->visualisation_Device, sizeof(Visualisation));

    calcLatticeSoundSpeed(config);
    calcRelaxationTime(config);
    //config->relaxationTime = 1;
    //config->cs = 1.0 / sqrt(3);
    printLGAConfig(config);

    createEmptySpace(config, config->nx, config->ny);
    //drawWall(config, 0, config->nx, 0, 1);
    //drawWall(config, 0, 1, 0, config->ny);
    //drawWall(config, config->nx - 1, 1, 0, config->ny);
    //drawWall(config, 0, config->nx, config->ny - 1, 1);
    //drawWall(config, config->nx / 5, 1, 1, config->ny / 3);
    //drawWall(config, config->nx / 5, 1, 2 * config->ny / 3, config->ny / 3);

 //   drawWall(config, config->nx / 2, 1, 0, config->ny / 5);
  //  drawWall(config, config->nx / 2, 1, 2*config->ny / 5, config->ny / 5);
  //  drawWall(config, config->nx / 2, 1, 4*config->ny / 5, config->ny/5);


    //randomInitialState(config, config->nx/5, config->nx / 5, config->ny/2, config->ny/4 - 1);
    //randomInitialState(config, 3*config->nx/5, config->nx / 5, config->ny/2, config->ny/4 - 1);
    //randomInitialState(config, 0, config->nx / 5 + 1, 1, config->ny-1);
   
    LBM_simulation(config);
    free(config->domain_Host);
    cudaFree(config->visualisation_Device);
    return 0;
}