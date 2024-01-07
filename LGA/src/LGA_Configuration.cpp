#include "..\inc\LGA_Configuration.h"
#include "device_launch_parameters.h"
#include"..\lib\glew-2.2.0\include\GL\glew.h"


void calculateBlockSize(LBM_Config* config)
{
	config->blockSize_Entire.x = 32;
	config->blockSize_Entire.y = 32;
	config->blockSize_Entire.z = 1;
	dim3 grid = { (config->nx + config->blockSize_Entire.x - 1) / config->blockSize_Entire.x,
			(config->ny + config->blockSize_Entire.y - 1) / config->blockSize_Entire.y, 1 };
	config->gridSize_Entire = grid;

	config->blockSize_Interior.x = 32;
	config->blockSize_Interior.y = 32;
	config->blockSize_Interior.z = 1;
	grid = { (config->nx - 2 + config->blockSize_Interior.x - 1) / config->blockSize_Interior.x,
			(config->ny - 2 + config->blockSize_Interior.y - 1) / config->blockSize_Interior.y, 1 };
	config->gridSize_Interior = grid;

	config->blockSize_BoundryE = 32;
	config->gridSize_BoundryE = (config->ny - 2 + config->blockSize_BoundryE - 1)/ config->blockSize_BoundryE;

	config->blockSize_BoundryW = 32;
	config->gridSize_BoundryW = (config->ny - 2 + config->blockSize_BoundryW - 1)/ config->blockSize_BoundryW;

	config->blockSize_BoundryN = 32;
	config->gridSize_BoundryN = (config->nx - 2 + config->blockSize_BoundryN - 1)/ config->blockSize_BoundryN;

	config->blockSize_BoundryS = 32;
	config->gridSize_BoundryS = (config->nx - 2 + config->blockSize_BoundryS - 1)/ config->blockSize_BoundryS;
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

