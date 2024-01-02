#pragma once

#include"LGA_kernels.cuh"
#include"..\inc\LGA_Graphics.h"


LBM_Config* LGA_setup(const char* setupFile);
void LBM_simulation(LBM_Config* configuration);

int createEmptySpace(LBM_Config* config, unsigned int nx, unsigned int ny);
void drawWall(LBM_Config* config, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height);

void randomInitialState(LBM_Config* config, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height);


