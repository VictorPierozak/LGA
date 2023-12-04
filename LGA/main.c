
#include<GL\glew.h>
#include<glfw3.h>
#include".\inc\LGA_Simulation.h"
int main()
{
    srand(time(NULL));
    cudaDeviceReset();
    LGA_Config* config = (LGA_Config*)malloc(sizeof(LGA_Config));
    config->domain_Device = NULL;
    config->domain_Host = NULL;
    config->isWorking = 0;
    config->shutDown = 0;

    createEmptySpace(config, 1024, 1024);
    drawWall(config, 0, config->nx, 0, 1);
    drawWall(config, 0, 1, 0, config->ny);
    drawWall(config, config->nx - 1, 1, 0, config->ny);
    drawWall(config, 0, config->nx, config->ny - 1, 1);
    drawWall(config, config->nx / 5, 1, 1, config->ny / 3);
    drawWall(config, config->nx / 5, 1, 2 * config->ny / 3, config->ny / 3);

    randomInitialState(config, 10e4, 0, config->nx / 5, 0, config->ny);
    
    LGA_simulation(config);
    free(config->domain_Host);
    return 0;
}