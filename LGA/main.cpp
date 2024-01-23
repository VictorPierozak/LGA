#include<iostream>
#include<GL\glew.h>
#include<glfw3.h>
#include".\inc\LGA_Simulation.h"
int main()
{
    srand(time(NULL));
   
    LBM_Config* config = (LBM_Config*)malloc(sizeof(LBM_Config));
    initLBM(&config, 0.1, 128, 128, FLOW, 0.1);

    config->dynamicViscousity = 1.8e-3;
    config->fluidRo = 1.2;
    config->defaultRo = 0.1; 1e-6;
    config->visualisation.field = 0;
    config->visualisation.amplifier = 1.0;
    config->visualisation.amplifierStep = 0.1;

    setBC_NormalVelocity(config, EAST, 0, -0.08);
    setBC_NormalVelocity(config, WEST, 0, 0.08);
    setBC_BouncyBack(config, SOUTH);
    setBC_Symmetry(config, NORTH);
    setBC_BouncyBack(config, NORTH);

    //setBC_ConstantVelocity(config, NORTH, 0, 0, 0.02, 0);
    //setBC_ConstantVelocity(config, SOUTH, 0, 0, 0, 0);
    //setBC_NormalVelocity(config, WEST, 0, 0.02);
    //setBC_NormalVelocity(config, EAST, 0, -0.02);
  
    cudaMalloc(&config->visualisation_Device, sizeof(Visualisation));

    calcLatticeSoundSpeed(config);
    calcRelaxationTime(config);
    printLGAConfig(config);

    createEmptySpace(config, config->nx, config->ny);

    //randomInitialState(config, config->nx/5, config->nx / 5, config->ny/5, config->ny/5 - 1);
    //randomInitialState(config, config->nx/5, config->nx / 5, 3*config->ny/5, config->ny/5 - 1);
    //randomInitialState(config, config->nx/5, 4*config->nx / 5, 2*config->ny/5, config->ny/5 - 1);
   
    LBM_simulation(config);
    free(config->domain_Host);
    cudaFree(config->visualisation_Device);
    return 0;
}