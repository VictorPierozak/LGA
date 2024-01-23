#pragma once

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"..\inc\CommonTypes.h"


typedef struct
{
	int field;
	float amplifier;
	float amplifierStep;
}Visualisation;

typedef struct
{
	int type;
	double vn0;
	double d_vn;
	double vt0;
	double d_vt;
	double density;
}BoundryCondition;

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

	dim3 blockSize_Interior;
	dim3 gridSize_Interior;

	dim3 blockSize_Entire;
	dim3 gridSize_Entire;

	int blockSize_BoundryN;
	int gridSize_BoundryN;
	int blockSize_BoundryS;
	int gridSize_BoundryS;
	int blockSize_BoundryE;
	int gridSize_BoundryE;
	int blockSize_BoundryW;
	int gridSize_BoundryW;

	// Simulation //

	int flow;

	double defaultRo;
	double v0[2];

	BoundryCondition northBC;
	BoundryCondition southBC;
	BoundryCondition westBC;
	BoundryCondition eastBC;

	Visualisation visualisation;
	Visualisation* visualisation_Device;

	int field;
	int isWorking;
	int shutDown;
	int doCopy; // 0 - no copy | 1 - do copy | 2 - done
} LBM_Config;

void calculateBlockSize(LBM_Config* config);
void pushDomainToDevice(LBM_Config* config);
void pullDomainFromDevice(LBM_Config* config);
void copyDomainFromDevice(LBM_Config* config);
void printLGAConfig(const LBM_Config* config);
void calcRelaxationTime(LBM_Config* config);
void calcLatticeSoundSpeed(LBM_Config* config);

void setBC_NormalVelocity(LBM_Config* config, int boundry, double normalVelocity);
void setBC_NormalVelocity(LBM_Config* config, int boundry, double v0, double v1);
void setBC_ConstantVelocity(LBM_Config* config, int boundry, double vn0, double vn1, double vt0, double vt1);
void setBC_BouncyBack(LBM_Config* config, int boundry);
void setBC_Symmetry(LBM_Config* config, int boundry);

void initLBM(LBM_Config** config, double length, int nx, int ny, int flow, double mach);

