#ifndef SIMULATION_H
#define SIMULATION_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <random>
#include <vector>
#include <glm/glm.hpp>
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
    // Stato e config
    SimulationState state;
    bool keys[1024]{};
    unsigned int width, height;

    // Agenti
    std::vector<Boid> boids;

    // Distanze per regole
    float cohesionDistance;
    float separationDistance;
    float alignmentDistance;
    float borderDistance;
    float predatorFearDistance;
    float predatorChaseDistance;
    float predatorSeparationDistance;

    // Pesi per regole
    float cohesionScale;
    float separationScale;
    float alignmentScale;
    float borderScale;
    float predatorFearScale;
    float predatorChaseScale;
    float predatorSeparationScale;

    // Costruttori / distruttore
    Simulation(unsigned int width, unsigned int height);
    ~Simulation();

    // API principali
    void init();
    void processInput(float dt);
    void update(float dt);
    void render();

private:
    // Renderer
    BoidRenderer* boidRender;

    // Boids (regole base)
    glm::vec2 moveTowardCenter(size_t i);
    glm::vec2 avoidNeighbors(size_t i);
    glm::vec2 matchVelocity(size_t i);

    // Preda/Predatore
    glm::vec2 evadePredators(size_t i);
    glm::vec2 chasePrey(size_t i);
    glm::vec2 avoidOtherPredators(size_t i);

	// Leader
    glm::vec2 followLeaders(size_t i);
    glm::vec2 leaderSeparation(size_t i);

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

    // Bordi
    glm::vec2 avoidBorders(const Boid& b);
};

#endif
