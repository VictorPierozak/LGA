#include "..\inc\LGA_Configuration.h"
#include "device_launch_parameters.h"
#include"..\lib\glew-2.2.0\include\GL\glew.h"


void calculateBlockSize(LGA_Config* config)
{
	config->blockSize.x = 32;
	config->blockSize.y = 32;
	config->blockSize.z = 1;
	dim3 grid = { (config->nx + config->blockSize.x - 1) / config->blockSize.x,
			(config->ny + config->blockSize.y - 1) / config->blockSize.y, 1 };
	config->gridSize = grid;
}

void pushDomainToDevice(LGA_Config* config)
{
	cudaMalloc(&config->domain_Device, sizeof(Field) * config->nx * config->ny);
	cudaMemcpy(config->domain_Device, config->domain_Host, sizeof(Field) * config->nx * config->ny, cudaMemcpyHostToDevice);
}

void pullDomainFromDevice(LGA_Config* config)
{
	cudaMemcpy(config->domain_Host, config->domain_Device, sizeof(Field) * config->nx * config->ny, cudaMemcpyDeviceToHost);
	cudaFree(config->domain_Device);
	config->domain_Device = NULL;
	cudaDeviceReset();
}

void copyDomainFromDevice(LGA_Config* config)
{
	cudaMemcpy(config->domain_Host, config->domain_Device, sizeof(Field) * config->nx * config->ny, cudaMemcpyDeviceToHost);
}



