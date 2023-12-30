#include<iostream>
#include<GL\glew.h>
#include<glfw3.h>
#include".\inc\LGA_Simulation.h"
int main()
{
    srand(time(NULL));
   
    LGA_Config* config = (LGA_Config*)malloc(sizeof(LGA_Config));
    config->domain_Device = NULL;
    config->domain_Host = NULL;
    config->isWorking = 0;
    config->shutDown = 0;
    config->nx = NX;
    config->ny = NY;

    config->dynamicViscousity = 1.81e-5;//0.0009;
    config->fluidRo = 1.293;
    config->dt = 0.1;
    config->dx = 0.1;
    config->defaultRo = 1e-6;
    config->field = 0;

    config->visualisation.field = 0;
    config->visualisation.amplifier = 1.0;
    config->visualisation.amplifierStep = 0.1;

    cudaMalloc(&config->visualisation_Device, sizeof(Visualisation));

    calcLatticeSoundSpeed(config);
    calcRelaxationTime(config);

    printLGAConfig(config);

    createEmptySpace(config, NX, NY);
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
   
    LGA_simulation(config);
    free(config->domain_Host);
    cudaFree(config->visualisation_Device);
    return 0;
}