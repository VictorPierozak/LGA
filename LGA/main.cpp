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
    config->nx = 128;
    config->ny = 128;

    config->dynamicViscousity = 1.867e-5;
    config->fluidRo = 1.286;
    config->dx = 2.0 /(double) config->nx; 0.1;
    config->dt = config->dx*0.4;
    config->defaultRo = 0; 1e-6;
    config->field = 0;

    config->visualisation.field = 0;
    config->visualisation.amplifier = 1.0;
    config->visualisation.amplifierStep = 0.1;

    cudaMalloc(&config->visualisation_Device, sizeof(Visualisation));

    calcLatticeSoundSpeed(config);
    calcRelaxationTime(config);
    //config->relaxationTime = 0.867;
    //config->cs = 1.0 / sqrt(3);
    printLGAConfig(config);

    createEmptySpace(config, config->nx, config->ny);
    drawWall(config, 0, config->nx, 0, 1);
    drawWall(config, 0, 1, 0, config->ny);
    drawWall(config, config->nx - 1, 1, 0, config->ny);
    drawWall(config, 0, config->nx, config->ny - 1, 1);
    drawWall(config, config->nx / 5, 1, 1, config->ny / 3);
    drawWall(config, config->nx / 5, 1, 2 * config->ny / 3, config->ny / 3);

 //   drawWall(config, config->nx / 2, 1, 0, config->ny / 5);
  //  drawWall(config, config->nx / 2, 1, 2*config->ny / 5, config->ny / 5);
  //  drawWall(config, config->nx / 2, 1, 4*config->ny / 5, config->ny/5);


    randomInitialState(config, 0, config->nx / 5 + 1, 0, config->ny);
   
    LBM_simulation(config);
    free(config->domain_Host);
    cudaFree(config->visualisation_Device);
    return 0;
}