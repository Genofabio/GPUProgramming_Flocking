#pragma once

#include <random>
#include <vector>

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
#include <gpu/BoidData.h>
#include <gpu/GridData.h>  // nuova struct per i buffer della griglia

class SimulationGPU
{
public:
    // Stato della simulazione
    bool keys[1024]{};
    unsigned int width, height;
    float currentTime = 0.0f;

    // Costruttori / distruttore
    SimulationGPU(unsigned int width, unsigned int height);
    ~SimulationGPU();

    void init();
    void processInput(float dt);
    void update(float dt);
    void render();

    // Profiling
    void updateStats(float dt);
    void saveProfilerCSV(const std::string& path);

private:
    // GPU data
    BoidData gpuBoids;
    bool boidDataInitialized = false;

    // GPU buffers per la uniform grid
    GridData gridData;   // struct che contiene tutti i buffer della griglia

    // Buffer GPU per il rendering
    glm::vec2* devRenderPositions = nullptr;
    float* devRenderRotations = nullptr;
    glm::vec3* devRenderColors = nullptr;
    float* devRenderScales = nullptr;

    std::vector<glm::vec2> renderPositions;
    std::vector<float> renderRotations;
    std::vector<glm::vec3> renderColors;
    std::vector<float> renderScales;

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
    void computeForces();
    void checkEatenPrey();
    void spawnNewBoids();

};
