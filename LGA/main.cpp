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

    config->simulationData.equalibriumStreams[0] = 0.25;
    config->simulationData.equalibriumStreams[1] = 0.25;
    config->simulationData.equalibriumStreams[2] = 0.25;
    config->simulationData.equalibriumStreams[3] = 0.25;
    config->simulationData.dt = 0.01;
    config->simulationData.tau = 1;

    createEmptySpace(config, NX, NY);
    drawWall(config, 0, config->nx, 0, 1);
    drawWall(config, 0, 1, 0, config->ny);
    drawWall(config, config->nx - 1, 1, 0, config->ny);
    drawWall(config, 0, config->nx, config->ny - 1, 1);
    drawWall(config, config->nx / 5, 1, 1, config->ny / 3);
    drawWall(config, config->nx / 5, 1, 2 * config->ny / 3, config->ny / 3);

    drawWall(config, config->nx / 2, 1, 0, config->ny / 5);
    drawWall(config, config->nx / 2, 1, 2*config->ny / 5, config->ny / 5);
    drawWall(config, config->nx / 2, 1, 4*config->ny / 5, config->ny/5);


    randomInitialState(config, 10e6, 1.0, 0, config->nx / 5 - 1, 0, config->ny);
    
    LGA_simulation(config);
    free(config->domain_Host);
    return 0;
}