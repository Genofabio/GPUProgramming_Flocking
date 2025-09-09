#ifndef SIMULATION_H
#define SIMULATION_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <random>
#include <vector>
#include <glm/glm.hpp>
#include "Boid.h"
#include "BoidRenderer.h"
#include "TextRenderer.h"
#include "Profiler.h"
#include "Wall.h"
#include "WallRenderer.h"
#include "Grid.h"
#include "GridRenderer.h"
#include "VectorRenderer.h"

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
    float currentTime = 0.0f;

    // Agenti
    std::vector<Boid> boids;
    std::vector<size_t> eatenPrey;
    std::vector<int> boidCouples;

    // Ostacoli
    Grid grid;
    std::vector<Wall> walls;

    // Parametri delle regole 
    // Distanze per regole
    float cohesionDistance;
    float separationDistance;
    float alignmentDistance;
    float borderDistance;
    float predatorFearDistance;
    float predatorChaseDistance;
    float predatorSeparationDistance;
    float predatorEatDistance;

    // Pesi per regole
    float cohesionScale;
    float separationScale;
    float alignmentScale;
    float borderScale;
    float predatorFearScale;
    float predatorChaseScale;
    float predatorSeparationScale;
    float borderAlertDistance;

	// Generatore di numeri casuali
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

    // Costruttori / distruttore
    Simulation(unsigned int width, unsigned int height);
    ~Simulation();

    // API principali
    void init();
    void processInput(float dt);
    void update(float dt);
    void render();

    // Funzioni principali con profiling
    void updateWithProfiling(float dt);
    void renderWithProfiling();
    void updateStats(float dt);
    void saveProfilerCSV(const std::string& path);

private:
    // Renderer
    BoidRenderer* boidRender;
    TextRenderer* textRender;
    WallRenderer* wallRender;
    GridRenderer* gridRender;
    VectorRenderer* vectorRender;

	// Profiler
    Profiler profiler;

    // Boids (regole base)
    glm::vec2 moveTowardCenter(size_t i);
    glm::vec2 avoidNeighbors(size_t i);
    glm::vec2 matchVelocity(size_t i);

    // Preda/Predatore
    glm::vec2 evadePredators(size_t i);
    glm::vec2 chasePrey(size_t i);
    void eatPrey(size_t predatorIndex, size_t preyIndex);
    glm::vec2 avoidOtherPredators(size_t i);

	// Leader
    glm::vec2 followLeaders(size_t i);
    glm::vec2 leaderSeparation(size_t i);

    // Crescita dei boids
    void upgradeBoid(Boid& b, float currentTime);

    // Spawn dei boids
    void updateMating();
    void spawnBoid(size_t parentA, size_t parentB);

    // Bordi
    glm::vec2 avoidBorders(size_t i);
    glm::vec2 avoidWalls(size_t i);
    glm::vec2 computeDrift(size_t i, float dt);

	// Utility
    std::vector<Wall> generateRandomWalls(int n);
};

#endif
