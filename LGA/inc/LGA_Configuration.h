#pragma once

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"..\inc\CommonTypes.h"

typedef struct
{
	float dt;
	float tau;
}Simulation_Data;

typedef struct
{
	int field;
	float amplifier;
	float amplifierStep;
}Visualisation;

typedef struct
{
	Field* domain_Host;
	Field* domain_Device;
	unsigned int nx;
	unsigned int ny;

	// //
	double dx;
	double dt;
	double fluidRo;
	double dynamicViscousity;
	double relaxationTime;
	double cs;

	// CUDA related variables //

	dim3 blockSize;
	dim3 gridSize;

	// Simulation //

	double defaultRo;
	double v0[2];

	Visualisation visualisation;
	Visualisation* visualisation_Device;

	int field;
	int isWorking;
	int shutDown;
} LGA_Config;

void calculateBlockSize(LGA_Config* config);
void pushDomainToDevice(LGA_Config* config);
void pullDomainFromDevice(LGA_Config* config);
void copyDomainFromDevice(LGA_Config* config);
void printLGAConfig(const LGA_Config* config);
void calcRelaxationTime(LGA_Config* config);
void calcLatticeSoundSpeed(LGA_Config* config);
