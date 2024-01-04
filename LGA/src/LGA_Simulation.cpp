#include"..\inc\LGA_Simulation.h"
#include<GL\glew.h>
#include<glfw3.h>
#include<stdio.h>
#include<Windows.h>
#include<ctime>
#include<stdio.h>
#include<omp.h>

// CPU THREADS //

CRITICAL_SECTION accessSimulationState;
CRITICAL_SECTION accessShutDown;
CRITICAL_SECTION accessCopy;
CRITICAL_SECTION accessWritingCmd;

int isStillWorking(LBM_Config* configuration)
{
	int result = 0;
	EnterCriticalSection(&accessSimulationState);
	result = configuration->isWorking;
	LeaveCriticalSection(&accessSimulationState);
	return result;
}

int shutDown(LBM_Config* configuration)
{
	int result = 0;
	EnterCriticalSection(&accessShutDown);
	result = configuration->shutDown;
	LeaveCriticalSection(&accessShutDown);
	return result;
}

int doCopy(LBM_Config* configuration)
{
	int result = 0;
	EnterCriticalSection(&accessCopy);
	result = configuration->doCopy;
	LeaveCriticalSection(&accessCopy);
	return result;
}

DWORD WINAPI RunSimulationParallel(LPVOID configuration)
{
	printf("\nSimulation start\n");
	LBM_Config* ptr = (LBM_Config*)configuration;
	//LGA_init(ptr);
	while (shutDown(ptr) == 0)
	{
		while (isStillWorking(ptr) == 0)
		{
			if (doCopy(ptr) == 1)
			{
				copyDomainFromDevice(ptr);
				EnterCriticalSection(&accessCopy);
				ptr->doCopy = 2;
				LeaveCriticalSection(&accessCopy);
			}
			else Sleep(1);
		}

		LBM_run(ptr);

		EnterCriticalSection(&accessSimulationState);
		ptr->isWorking = 0;
		LeaveCriticalSection(&accessSimulationState);
	}
	ptr->shutDown = -1;
	printf("\nSimulation shut down\n");
	return 0;
}

DWORD WINAPI calculateTotalMass(LPVOID ptr)
{
	LBM_Config* configuration = (LBM_Config*)ptr;
	EnterCriticalSection(&accessCopy);
	configuration->doCopy = 1;
	LeaveCriticalSection(&accessCopy);

	while (doCopy(configuration) != 2) Sleep(10);

	EnterCriticalSection(&accessCopy);
	configuration->doCopy = 0;
	LeaveCriticalSection(&accessCopy);

	double sum = 0;
	long int size = configuration->nx * configuration->ny;
//#pragma omp parallel for reduction(+:sum) default(none) firstprivate(size) num_threads(8)
	for (long int i = 0; i < size; i++)
		sum += configuration->domain_Host[i].ro;
	printf("=== TOTAL MASS: %lf ===\n", sum);
	return 0;
}


//

LBM_Config* LGA_setup(const char* setupFile)
{
	return NULL;
}

int createEmptySpace(LBM_Config* config, unsigned int nx, unsigned int ny)
{
	if (config == NULL)
	{
		return -1;
	}

	if (config->domain_Device != nullptr)
	{
		return -1;
	}

	if (config->domain_Host != nullptr)
	{
		free(config->domain_Host);
		config->domain_Host = NULL;
	}
	
	config->domain_Host = (Field*)malloc(sizeof(Field) * nx * ny);
	config->nx = nx;
	config->ny = ny;
	for (unsigned int y = 0; y < ny; y++)
	{
		for (unsigned int x = 0; x < nx; x++)
		{
			config->domain_Host[x + y * nx].type = EMPTY_SPACE;
			config->domain_Host[x + y * nx].ro = config->defaultRo;
			config->domain_Host[x + y * nx].u[0] = 0;
			config->domain_Host[x + y * nx].u[1] = 0;
			for (int i = 0; i < 9; i++)
			{
				config->domain_Host[x + y * nx].inStreams[i] = 0;
				config->domain_Host[x + y * nx].outStreams[i] = 0;
				config->domain_Host[x + y * nx].eqStreams[i] = 0;
			}
			
		}
	}
	return 0;
}


void drawWall(LBM_Config* config, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height)
{
	unsigned int yend = y0 + height;
	unsigned int xend = x0 + width;
	for (unsigned int y = y0; y < yend; y++)
	{
		for (unsigned int x = x0; x < xend; x++)
		{
			config->domain_Host[x + y * config->nx].type = WALL;
		}
	}
}

void keyEvents(GLFWwindow* window, LBM_Config* configuration)
{
	if (glfwGetKey(window, '1') == GLFW_PRESS) {
		//EnterCriticalSection(&accessSimulationState);
		configuration->visualisation.field = 0;
		//LeaveCriticalSection(&accessSimulationState);
	}
	if (glfwGetKey(window, '2') == GLFW_PRESS) {
		//EnterCriticalSection(&accessSimulationState);
		configuration->visualisation.field = 1;
		//LeaveCriticalSection(&accessSimulationState);
	}
	if (glfwGetKey(window, '3') == GLFW_PRESS) {
		//EnterCriticalSection(&accessSimulationState);
		configuration->visualisation.field = 2;
		//LeaveCriticalSection(&accessSimulationState);
	}
	if (glfwGetKey(window, '4') == GLFW_PRESS) {
		//EnterCriticalSection(&accessSimulationState);
		configuration->visualisation.field = 3;
		//LeaveCriticalSection(&accessSimulationState);
	}
	if (glfwGetKey(window, 'M') == GLFW_PRESS) {
		HANDLE totalMass = NULL;
		DWORD pid_simulation;

		// Thread creation //
		totalMass = CreateThread(NULL, 0, calculateTotalMass, configuration, 0, &pid_simulation);
		// Detach //
		CloseHandle(totalMass);
	}
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		printf("=== AMPLIFIER ===\n Previouse: %f\t|\t", configuration->visualisation.amplifier);
		configuration->visualisation.amplifier += configuration->visualisation.amplifierStep;
		printf("New: %f\n", configuration->visualisation.amplifier);
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		printf("=== AMPLIFIER ===\n Previouse: %f\t|\t", configuration->visualisation.amplifier);
		configuration->visualisation.amplifier -= configuration->visualisation.amplifierStep;
		printf("New: %f\n", configuration->visualisation.amplifier);
	}
}

void LBM_simulation(LBM_Config* configuration)
{
	// Set GLFW error callback
	// Configure GLFW
	if (!glfwInit()) {
		printf("Failed to initialize GLFW");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a window
	int width = 800;
	int height = 600;
	const char* title = "LGA";

	GLenum err = glewInit();
	if (err != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		// Handle GLEW initialization error
	}

	GLFWwindow* window = glfwCreateWindow(width, height, title, nullptr, nullptr);
	if (!window) {
		printf("\nFailed to create GLFW window\n");
		glfwTerminate();
		return;
	}
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Set up //
	calculateBlockSize(configuration);
	pushDomainToDevice(configuration); 
	setConstantMemory(configuration);

	// Graphics resources set up //
	Graphics_Objects* graphicsRes = createGraphicsObjects(configuration);

	// Threads WINAPI //
	HANDLE simulation = NULL;
	DWORD pid_simulation;
	InitializeCriticalSection(&accessSimulationState);
	InitializeCriticalSection(&accessShutDown);
	InitializeCriticalSection(&accessCopy);

	// Thread creation //
	simulation = CreateThread(NULL, 0, RunSimulationParallel, configuration, 0, &pid_simulation);
	// Detach //
	CloseHandle(simulation);
	//Sleep(10000);
	// Main loop
	while (!glfwWindowShouldClose(window)) {
		// Process input
		glfwPollEvents();
		keyEvents(window, configuration);
		if(isStillWorking(configuration) == 0)
		{
			// Update VBO //
			mapCudaResources(graphicsRes);
			LBM_draw(configuration, graphicsRes->devPtr);
			unmapCudaGraphicResources(graphicsRes);
			// Start new work //
			EnterCriticalSection(&accessSimulationState);
			configuration->isWorking = 1;
			LeaveCriticalSection(&accessSimulationState);
		}

		// Swap front and back buffers
		glClearColor(0.0f, 0.4f, 0.5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//glDrawArrays(GL_TRIANGLES, 0, 4*configuration->nx*configuration->ny);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, graphicsRes->ebo);
		glDrawElements(GL_TRIANGLES, configuration->nx * configuration->ny * 6, GL_UNSIGNED_INT, 0);
		glfwSwapBuffers(window);
	}
	EnterCriticalSection(&accessShutDown);
	configuration->shutDown = 1;
	LeaveCriticalSection(&accessShutDown);

	// Terminate GLFW
	glfwTerminate();

	while (shutDown(configuration) != -1)Sleep(10);
	// Close critical section //
	DeleteCriticalSection(&accessSimulationState);
	DeleteCriticalSection(&accessShutDown);

	// Release graphic resources //
	//releaseCudaGraphicResources(graphicsRes);
	releaseOpenGLResources(graphicsRes);
	pullDomainFromDevice(configuration);
	free(graphicsRes);
}

void randomInitialState(LBM_Config* config, unsigned int x0, unsigned int width, unsigned int y0, unsigned int height)
{
	unsigned int counter = 0;
	unsigned int size = config->nx * config->ny;
	for(unsigned int y = y0; y < y0 + height; y++)
	for(unsigned int x = x0; x < x0 + width; x++)
	{ 
		int idx = x + y*config->nx; 
		if (config->domain_Host[idx].type == WALL) continue;

		config->domain_Host[idx].ro = (float)rand() / (float)RAND_MAX ;

	}
}