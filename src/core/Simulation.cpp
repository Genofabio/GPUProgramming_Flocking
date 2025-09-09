#include <set>
#include <map>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

#include <core/Simulation.h>
#include <utility/ResourceManager.h>

Simulation::Simulation(unsigned int width, unsigned int height)
    : keys()
    , width(width)
    , height(height)
    , grid(10, 15, static_cast<float>(width), static_cast<float>(height))
    , boidRenderer(nullptr)
    , wallRenderer(nullptr)
    , gridRenderer(nullptr)
    , textRenderer(nullptr)
    , rng(std::random_device{}())
    , dist(-1.0f, 1.0f)
{
    // Inizializzazione dei parametri di boid
    params.maxSpeed = 100.0f;
    params.slowDownFactor = 0.3f;

    params.cohesionDistance = 100.0f;
    params.separationDistance = 25.0f;
    params.alignmentDistance = 50.0f;
    params.borderDistance = 300.0f;
    params.predatorFearDistance = 150.0f;
    params.predatorChaseDistance = 800.0f;
    params.predatorSeparationDistance = 60.0f;
    params.predatorEatDistance = 5.0f;

    params.cohesionScale = 0.05f;
    params.separationScale = 2.0f;
    params.alignmentScale = 0.15f;
    params.borderScale = 0.3f;
    params.predatorFearScale = 0.8f;
    params.predatorChaseScale = 0.12f;
    params.predatorSeparationScale = 2.0f;
    params.borderAlertDistance = (static_cast<float>(height) / 5.0f);

    params.leaderInfluenceDistance = 200.0f;
    params.mateDistance = 10.0f;
    params.mateThreshold = 200;
    params.matingAge = 6;
    params.predatorBoostRadius = 80.0f;
    params.desiredLeaderDistance = 200.0f;
}

Simulation::~Simulation()
{
    delete wallRenderer;
    delete boidRenderer;
    delete textRenderer;
    delete gridRenderer;
}

void Simulation::init()
{
    // Shader setup
    ResourceManager::LoadShader("shaders/boid_shader.vert", "shaders/boid_shader.frag", nullptr, "boid");
    ResourceManager::LoadShader("shaders/wall_shader.vert", "shaders/wall_shader.frag", nullptr, "wall");
    ResourceManager::LoadShader("shaders/grid_shader.vert", "shaders/grid_shader.frag", nullptr, "grid");
    ResourceManager::LoadShader("shaders/text_shader.vert", "shaders/text_shader.frag", nullptr, "text");

    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -1.0f, 1.0f);

    ResourceManager::GetShader("boid").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("wall").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("grid").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("text").Use().SetInteger("text", 0);
    ResourceManager::GetShader("text").SetMatrix4("projection", projection);

    // Renderer setup
    boidRenderer = new BoidRenderer(ResourceManager::GetShader("boid"));
    wallRenderer = new WallRenderer(ResourceManager::GetShader("wall"));
    gridRenderer = new GridRenderer(ResourceManager::GetShader("grid"));
    textRenderer = new TextRenderer(ResourceManager::GetShader("text"));
    textRenderer->loadFont("resources/fonts/Roboto/Roboto-Regular.ttf", 24);

    // Initialize boids
    initLeaders(0);
    initPrey(180);
    initPredators(2);

    // Initialize walls
    initWalls(50);
}

void Simulation::update(float dt)
{
    profiler.start();
    currentTime += dt;

    size_t N = boids.size();
    std::vector<glm::vec2> velocityChanges(N, glm::vec2(0.0f));

    // 1. Calcola tutte le forze
    computeForces(velocityChanges);

    // 2. Applica le velocità ai boid
    applyVelocity(dt, velocityChanges);

    // 3. Controlla quali prede sono state mangiate e le rimuove
    checkEatenPrey();

    // 4. Gestisce l'accoppiamento e lo spawn di nuovi boid
    spawnNewBoids();

    profiler.log("update", profiler.stop());
}

void Simulation::render()
{
    profiler.start();

    // 1. Draw grid
    glLineWidth(1.0f);
    gridRenderer->draw(grid, glm::vec3(0.2f, 0.2f, 0.2f));

    // 2. Draw boids
    for (const Boid& b : boids) {
        float angle = glm::degrees(atan2(b.velocity.y, b.velocity.x)) + 270.0f;
        boidRenderer->draw(b.position, angle, b.color, 10.0f * b.scale);
    }

    // 3. Draw walls
    glLineWidth(3.0f);
    for (const Wall& w : walls)
        wallRenderer->draw(w, glm::vec3(0.25f, 0.88f, 0.82f));

    profiler.log("render", profiler.stop());

    // 4. Draw HUD / stats
    float margin = 10.0f;
    float scale = 0.7f;
    glm::vec3 color(0.9f, 0.9f, 0.3f);

    double fps = profiler.getCurrentFPS();
    if (fps > 0.0)
        textRenderer->draw("FPS: " + std::to_string(static_cast<int>(fps)), margin, height - margin - 20.0f, scale, color);

    textRenderer->draw("Boids: " + std::to_string(boids.size()), margin, height - margin - 40.0f, scale, color);
}

void Simulation::processInput(float dt) {}

void Simulation::updateStats(float dt) {
    profiler.updateFrameStats(dt);
}

void Simulation::saveProfilerCSV(const std::string& path) {
    profiler.saveCSV(path);
}


// === HELPER Init ===
void Simulation::initLeaders(int count)
{
    for (int i = 0; i < count; ++i) {
        Boid b;
        b.position = glm::vec2(static_cast<float>(rand() % width), static_cast<float>(rand() % height));
        b.velocity = glm::vec2(static_cast<float>(rand() % 201 - 100) * 0.5f, static_cast<float>(rand() % 201 - 100) * 0.5f);
        b.type = BoidType::LEADER;
        b.age = 10;
        b.scale = 1.7f;
        b.color = glm::vec3(0.9f, 0.9f, 0.2f);
        b.drift = glm::vec2(0);
        boids.push_back(b);
    }
}

void Simulation::initPrey(int count)
{
    std::uniform_int_distribution<int> ageDist(0, 6);
    std::uniform_real_distribution<float> offsetDist(0.0f, 30.0f);

    for (int i = 0; i < count; ++i) {
        Boid b;
        b.position = glm::vec2(static_cast<float>(rand() % width), static_cast<float>(rand() % height));
        b.velocity = glm::vec2(static_cast<float>(rand() % 201 - 100) * 0.5f, static_cast<float>(rand() % 201 - 100) * 0.5f);
        b.type = BoidType::PREY;
        b.birthTime = currentTime + offsetDist(rng);
        b.age = ageDist(rng);
        float t = b.age / 10.0f;
        b.scale = 1.0f + 0.04f * b.age;
        b.color = glm::mix(glm::vec3(0.2f, 0.2f, 0.9f), glm::vec3(0.05f, 0.8f, 0.7f), t);
        b.influence = 0.8f + 0.04f * b.age;
        boids.push_back(b);
    }
}

void Simulation::initPredators(int count)
{
    for (int i = 0; i < count; ++i) {
        Boid b;
        b.position = glm::vec2(static_cast<float>(rand() % width), static_cast<float>(rand() % height));
        b.velocity = glm::vec2(static_cast<float>(rand() % 201 - 100) * 0.5f, static_cast<float>(rand() % 201 - 100) * 0.5f);
        b.type = BoidType::PREDATOR;
        b.age = 10;
        b.scale = 1.9f;
        b.color = glm::vec3(0.9f, 0.2f, 0.2f);
        b.drift = glm::vec2(0);
        boids.push_back(b);
    }
}

void Simulation::initWalls(int count)
{
    auto candidates = grid.cellEdges;
    std::shuffle(candidates.begin(), candidates.end(), rng);

    std::set<std::pair<std::pair<int, int>, std::pair<int, int>>> usedEdges;
    std::map<std::pair<int, int>, std::pair<bool, bool>> vertexOccupancy;

    int added = 0;
    for (const auto& edge : candidates) {
        if (added >= count) break;

        auto p1 = std::make_pair(int(edge.first.x), int(edge.first.y));
        auto p2 = std::make_pair(int(edge.second.x), int(edge.second.y));
        if (p2 < p1) std::swap(p1, p2);

        bool horizontal = (p1.second == p2.second);
        if (usedEdges.count({ p1, p2 })) continue;
        if ((horizontal && (vertexOccupancy[p1].second || vertexOccupancy[p2].second)) ||
            (!horizontal && (vertexOccupancy[p1].first || vertexOccupancy[p2].first))) continue;

        Wall w({ edge.first, edge.second }, height / 11.0f, 3.0f);
        walls.push_back(w);  // popola direttamente il membro della classe

        usedEdges.insert({ p1, p2 });
        if (horizontal) {
            vertexOccupancy[p1].first = true;
            vertexOccupancy[p2].first = true;
        }
        else {
            vertexOccupancy[p1].second = true;
            vertexOccupancy[p2].second = true;
        }

        added++;
    }
}

// === HELPER Update ===
void Simulation::computeForces(std::vector<glm::vec2>& velocityChanges)
{
    size_t N = boids.size();

    for (size_t i = 0; i < N; ++i) {
        Boid& b = boids[i];
        glm::vec2 totalChange(0.0f);
        BoidRules::computeBoidUpgrade(b, currentTime);
        switch (b.type) {
        case PREY:
            totalChange += BoidRules::computeCohesion(b, boids, params.cohesionDistance, params.cohesionScale);
            totalChange += BoidRules::computeSeparation(b, boids, params.separationDistance, params.separationScale);
            totalChange += BoidRules::computeAlignment(b, boids, params.alignmentDistance, params.alignmentScale);
            totalChange += BoidRules::computeFollowLeaders(b, boids.data(), N, params.leaderInfluenceDistance);
            totalChange += BoidRules::computeBorderRepulsion(b.position, static_cast<float>(width), static_cast<float>(height), params.borderAlertDistance);
            totalChange += BoidRules::computeWallRepulsion(b.position, b.velocity, walls);
            break;

        case PREDATOR:
            totalChange += BoidRules::computeChasePrey(i, boids, params.predatorChaseDistance, params.predatorChaseScale, params.predatorBoostRadius);
            totalChange += BoidRules::computePredatorSeparation(b, boids, params.predatorSeparationDistance) * params.predatorSeparationScale;
            totalChange += BoidRules::computeBorderRepulsion(b.position, static_cast<float>(width), static_cast<float>(height), params.borderAlertDistance);
            totalChange += BoidRules::computeWallRepulsion(b.position, b.velocity, walls);
            break;

        case LEADER:
            totalChange += BoidRules::computeLeaderSeparation(b, boids.data(), N, params.desiredLeaderDistance);
            totalChange += BoidRules::computeBorderRepulsion(b.position, static_cast<float>(width), static_cast<float>(height), params.borderAlertDistance);
            totalChange += BoidRules::computeWallRepulsion(b.position, b.velocity, walls);
            break;
        }

        velocityChanges[i] = totalChange;
    }
}

void Simulation::applyVelocity(float dt, std::vector<glm::vec2>& velocityChanges)
{
    size_t N = boids.size();
    for (size_t i = 0; i < N; ++i) {
        Boid& b = boids[i];
        b.velocity += velocityChanges[i] * params.slowDownFactor;
        float speed = glm::length(b.velocity);
        if (speed > params.maxSpeed) b.velocity = (b.velocity / speed) * params.maxSpeed;
        b.position += b.velocity * dt;
    }
}

void Simulation::checkEatenPrey()
{
    size_t N = boids.size();
    std::vector<size_t> eatenPreyLocal;

    for (size_t i = 0; i < N; ++i) {
        if (boids[i].type != PREDATOR) continue;
        for (size_t j = 0; j < N; ++j) {
            if (boids[j].type != PREY) continue;
            if (BoidRules::computeEatPrey(i, j, boids, params.predatorEatDistance))
                eatenPreyLocal.push_back(j);
        }
    }

    // Rimuove le prede mangiate
    if (!eatenPreyLocal.empty()) {
        std::sort(eatenPreyLocal.rbegin(), eatenPreyLocal.rend());
        for (size_t idx : eatenPreyLocal) {
            if (idx < boids.size() && boids[idx].type == PREY)
                boids.erase(boids.begin() + idx);
        }
    }
}

void Simulation::spawnNewBoids()
{
    std::vector<std::pair<size_t, size_t>> spawnPairs;
    std::vector<int> boidCouplesLocal;

    size_t N = boids.size();
    for (size_t i = 0; i < N; ++i) {
        if (boids[i].type != PREY) continue;
        BoidRules::computeMating(i, boids, boidCouplesLocal, spawnPairs, params.mateDistance, params.mateThreshold, params.matingAge);
    }

    // Spawn nuovi boid
    for (auto& p : spawnPairs)
        boids.push_back(BoidRules::computeSpawnedBoid(boids[p.first], boids[p.second], currentTime));
}