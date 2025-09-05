#ifndef SIMULATION_H
#define SIMULATION_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <random>

#include <vector>
#include "Boid.h"
#include "BoidRenderer.h"
#include "Wall.h"
#include "WallRenderer.h"
#include "Grid.h"
#include "GridRenderer.h"

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

    // Griglia
    Grid grid;

    // Lista degli ostacoli
    std::vector<Wall> walls;
    std::vector<glm::vec2> corners;

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
    WallRenderer* wallRender;
    GridRenderer* gridRender;

    // Funzioni interne per calcolare i contributi delle regole
    glm::vec2 moveTowardCenter(size_t i);
    glm::vec2 avoidNeighbors(size_t i);
    glm::vec2 matchVelocity(size_t i);
    glm::vec2 avoidBorders(size_t i);
    glm::vec2 avoidWalls(size_t i);
    glm::vec2 avoidCorners(size_t i);

    float pointSegmentDistance(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, glm::vec2& closest);
    std::vector<Wall> generateRandomWalls(int n);
    std::vector<glm::vec2> computeWallCorners();
};
#endif