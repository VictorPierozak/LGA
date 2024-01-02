#include "..\inc\LGA_Configuration.h"
#include "device_launch_parameters.h"
#include"..\lib\glew-2.2.0\include\GL\glew.h"


void calculateBlockSize(LBM_Config* config)
{
	config->blockSize.x = 32;
	config->blockSize.y = 32;
	config->blockSize.z = 1;
	dim3 grid = { (config->nx + config->blockSize.x - 1) / config->blockSize.x,
			(config->ny + config->blockSize.y - 1) / config->blockSize.y, 1 };
	config->gridSize = grid;
}

void pushDomainToDevice(LBM_Config* config)
{
	cudaMalloc(&config->domain_Device, sizeof(Field) * config->nx * config->ny);
	cudaMemcpy(config->domain_Device, config->domain_Host, sizeof(Field) * config->nx * config->ny, cudaMemcpyHostToDevice);
}

void pullDomainFromDevice(LBM_Config* config)
{
	cudaMemcpy(config->domain_Host, config->domain_Device, sizeof(Field) * config->nx * config->ny, cudaMemcpyDeviceToHost);
	cudaFree(config->domain_Device);
	config->domain_Device = NULL;
	cudaDeviceReset();
}

void copyDomainFromDevice(LBM_Config* config)
{
	cudaMemcpy(config->domain_Host, config->domain_Device, sizeof(Field) * config->nx * config->ny, cudaMemcpyDeviceToHost);
}

void printLGAConfig(const LBM_Config* config) {
    printf("LGA_Config Data:\n");
    printf("nx: %u\n", config->nx);
    printf("ny: %u\n", config->ny);
    printf("dx: %lf\n", config->dx);
    printf("dt: %lf\n", config->dt);
    printf("defaultRo: %lf\n", config->defaultRo);
    //printf("fluidRo: %lf\n", config->fluidRo);
    //printf("dynamicViscosity: %lf\n", config->dynamicViscousity);
    printf("relaxationTime: %lf\n", config->relaxationTime);
    printf("cs: %lf\n", config->cs);
}

void calcRelaxationTime(LBM_Config* config)
{
	config->relaxationTime = config->dynamicViscousity / (config->fluidRo * config->cs* config->cs* config->dt) + 0.5;
}

void calcLatticeSoundSpeed(LBM_Config* config)
{
	config->cs = config->dx / (config->dt * sqrt(3));
}

