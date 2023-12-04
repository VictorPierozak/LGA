#pragma once

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"..\inc\CommonTypes.h"

typedef struct 
{
	Field* domain_Host;
	Field* domain_Device;
	unsigned int nx;
	unsigned int ny;

	// CUDA related variables //

	dim3 blockSize;
	dim3 gridSize;

	// Simulation //
	int isWorking;
	int shutDown;
} LGA_Config;

void calculateBlockSize(LGA_Config* config);
void pushDomainToDevice(LGA_Config* config);
void pullDomainFromDevice(LGA_Config* config);
void copyDomainFromDevice(LGA_Config* config);