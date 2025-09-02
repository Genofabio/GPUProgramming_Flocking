#ifndef SIMULATION_H
#define SIMULATION_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>

enum SimulationState {
    SIMULATION_RUNNING,
    SIMULATION_PAUSED,
    SIMULATION_FINISHED
};

class Simulation
{
public:
    SimulationState State;
    bool Keys[1024];
    unsigned int Width, Height;

    Simulation(unsigned int width, unsigned int height);
    ~Simulation();

    void Init();
    void ProcessInput(float dt);
    void Update(float dt);
    void Render();
};
#endif