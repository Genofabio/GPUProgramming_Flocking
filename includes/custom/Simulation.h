#ifndef SIMULATION_H
#define SIMULATION_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <random>

#include <vector>
#include "Boid.h"
#include "BoidRenderer.h"

enum SimulationState {
    SIMULATION_RUNNING,
    SIMULATION_PAUSED,
    SIMULATION_FINISHED
};

class Simulation
{
public:
    // Stato generale e configurazione base
    SimulationState state;
    bool keys[1024];
    unsigned int width, height;

    // Lista dei boids
    std::vector<Boid> boids;

    // Parametri delle regole 
    float cohesionDistance;
    float separationDistance;
    float alignmentDistance;
    float cohesionScale;
    float separationScale;
    float alignmentScale;
    float borderAlertDistance;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

    // Costruttori / distruttori
    Simulation(unsigned int width, unsigned int height);
    ~Simulation();

    // Funzioni principali
    void init();                  // inizializza boids, shaders, textures
    void processInput(float dt);  // gestisce input tastiera
    void update(float dt);        // aggiorna la simulazione (flocking)
    void render();                // disegna i boids

private:
    // Renderers
    BoidRenderer* boidRender;

    // Funzioni interne per calcolare i contributi delle regole
    glm::vec2 moveTowardCenter(size_t i);
    glm::vec2 avoidNeighbors(size_t i);
    glm::vec2 matchVelocity(size_t i);
    glm::vec2 avoidBordersSmooth(const Boid& b, float width, float height, float borderAlertDistance, float borderScale);
};
#endif