#pragma once

#include <random>

#include <external/glad/glad.h>
#include <external/GLFW/glfw3.h>

#include <graphics/BoidRenderer.h>
#include <graphics/TextRenderer.h>
#include <graphics/WallRenderer.h>
#include <graphics/GridRenderer.h>
#include <graphics/VectorRenderer.h>

#include <core/Profiler.h>
#include <core/BoidRules.h>
#include <core/Grid.h>
#include <core/UniformBoidGrid.h>
#include <core/BoidParams.h>

class Simulation
{
public:
    // Stato della simulazione
    bool keys[1024]{};
    unsigned int width, height;
    float currentTime = 0.0f;

    // Costruttori / distruttore
    Simulation(unsigned int width, unsigned int height);
    ~Simulation();

    void init();
    void processInput(float dt);
    void update(float dt);
    void render();

    // Profiling
    void updateStats(float dt);
    void saveProfilerCSV(const std::string& path);

private:
    // Agenti e ostacoli
    std::vector<Boid> boids;
    std::vector<Wall> walls;

    // Griglie
    Grid wallGrid;
    UniformBoidGrid boidGrid;


    // Parametri di simulazione
    BoidParams params;

    // RNG
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;

    // Renderers
    BoidRenderer* boidRenderer = nullptr;
    TextRenderer* textRenderer = nullptr;
    WallRenderer* wallRenderer = nullptr;
    GridRenderer* gridRenderer = nullptr;
    VectorRenderer* vectorRenderer = nullptr;

    // Profiler
    Profiler profiler;

    // Helper init
    void initLeaders(int count);
    void initPrey(int count);
    void initPredators(int count);
    void initWalls(int n);

    // Helper update
    void computeForces(std::vector<glm::vec2>& velocityChanges);
    void applyVelocity(float dt, std::vector<glm::vec2>& velocityChanges);
    void checkEatenPrey();
    void spawnNewBoids();
};
