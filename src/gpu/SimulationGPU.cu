#include <set>
#include <map>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>

#include <gpu/SimulationGPU.cuh>
#include <utility/ResourceManager.h>
#include <gpu/BoidData.h>
#include <core/Boid.h>
#include <gpu/cuda_kernel.cuh>
#include <cuda_runtime.h>

SimulationGPU::SimulationGPU(unsigned int width, unsigned int height)
    : keys()
    , width(width)
    , height(height)
    , wallGrid(10, 15, static_cast<float>(width), static_cast<float>(height), 1)
    , boidRenderer(nullptr)
    , wallRenderer(nullptr)
    , gridRenderer(nullptr)
    , textRenderer(nullptr)
    , rng(std::random_device{}())
    , dist(-1.0f, 1.0f)
{
    // Inizializzazione dei parametri di boid
    params.maxSpeed = 100.0f;          // come prima, velocità naturale
    params.slowDownFactor = 0.3f;      // frenata normale

    // Distanze (adattate a maxBoidDistance = 150)
    params.cohesionDistance = 60.0f;   // leggermente più piccolo per stare entro griglia
    params.separationDistance = 20.0f; // mantenere sicurezza nelle collisioni
    params.alignmentDistance = 40.0f;  // simile all’originale ma compatto
    params.borderDistance = 80.0f;     // non serve così grande
    params.predatorFearDistance = 100.0f;
    params.predatorChaseDistance = 120.0f;
    params.predatorSeparationDistance = 50.0f;
    params.predatorEatDistance = 5.0f; // come prima

    // Scale (forza delle regole)
    params.cohesionScale = 0.1f;      // originale
    params.separationScale = 2.2f;     // come prima
    params.alignmentScale = 0.19f;     // originale
    params.borderScale = 0.3f;         // coerente
    params.predatorFearScale = 0.8f;   // originale
    params.predatorChaseScale = 0.12f;
    params.predatorSeparationScale = 2.0f;
    params.borderAlertDistance = height / 5.0f;

    // Social/extra
    params.leaderInfluenceDistance = 120.0f; // ridotto proporzionalmente
    params.leaderInfluenceScale = 1.0f;
    params.mateDistance = 10.0f;
    params.mateThreshold = 200;
    params.matingAge = 6;
    params.predatorBoostRadius = 70.0f;
    params.desiredLeaderDistance = 150.0f;
    params.allyRadius = 50.0f;

    // Griglia boid basata sulla distanza massima di interazione
    float maxBoidDistance = 2 * std::max({ params.cohesionDistance, params.separationDistance, params.alignmentDistance, params.borderDistance, params.predatorFearDistance, params.predatorChaseDistance, params.predatorSeparationDistance, params.predatorEatDistance });
    int nCols = static_cast<int>(std::ceil(width / maxBoidDistance));
    int nRows = static_cast<int>(std::ceil(height / maxBoidDistance));
    boidGrid = UniformBoidGrid(nRows, nCols, static_cast<float>(width), static_cast<float>(height));
}

SimulationGPU::~SimulationGPU()
{
    delete wallRenderer;
    delete boidRenderer;
    delete textRenderer;
    delete gridRenderer;
    delete vectorRenderer;
}

void SimulationGPU::init()
{
    // Shader setup
    ResourceManager::LoadShader("shaders/gpu_boid_shader.vert", "shaders/gpu_boid_shader.frag", nullptr, "boid");
    ResourceManager::LoadShader("shaders/wall_shader.vert", "shaders/wall_shader.frag", nullptr, "wall");
    ResourceManager::LoadShader("shaders/grid_shader.vert", "shaders/grid_shader.frag", nullptr, "grid");
    ResourceManager::LoadShader("shaders/text_shader.vert", "shaders/text_shader.frag", nullptr, "text");
    ResourceManager::LoadShader("shaders/vector_shader.vert", "shaders/vector_shader.frag", nullptr, "vector");

    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -1.0f, 1.0f);

    ResourceManager::GetShader("boid").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("wall").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("grid").Use().SetMatrix4("projection", projection);
    ResourceManager::GetShader("text").Use().SetInteger("text", 0);
    ResourceManager::GetShader("text").SetMatrix4("projection", projection);
    ResourceManager::GetShader("vector").Use().SetMatrix4("projection", projection);

    // Renderer setup
    boidRenderer = new BoidRenderer(ResourceManager::GetShader("boid"));
    wallRenderer = new WallRenderer(ResourceManager::GetShader("wall"));
    gridRenderer = new GridRenderer(ResourceManager::GetShader("grid"));
    textRenderer = new TextRenderer(ResourceManager::GetShader("text"));
    textRenderer->loadFont("resources/fonts/Roboto/Roboto-Regular.ttf", 24);
    vectorRenderer = new VectorRenderer(ResourceManager::GetShader("vector"));

    // Initialize boids
    initLeaders(0);
    initPrey(20);
    initPredators(0);

    // Allocate and copy boid data to GPU
    if (!boidDataInitialized) {
        allocateBoidDataGPU(gpuBoids, boids.size());
        copyBoidsToGPU(boids, gpuBoids);
        boidDataInitialized = true;
    }

    // Initialize walls
    initWalls(50);
}

void SimulationGPU::update(float dt)
{
    profiler.start();
    currentTime += dt;

    // Aggiorna griglia CPU
    std::vector<glm::vec2> positions;
    for (const auto& b : boids)
        positions.push_back(b.position);

    boidGrid.updateCells(positions);

    size_t N = boids.size();
    std::vector<glm::vec2> velocityChanges(N, glm::vec2(0.0f));

    // Aggiorna GPU con i dati CPU

    // 1. Calcola tutte le forze sul GPU
    computeForces(velocityChanges);

    // 2. Applica le velocità ai boid CPU
    applyVelocity(dt, velocityChanges);

    // 3. Controlla quali prede sono state mangiate e le rimuove
    //checkEatenPrey();

    // 4. Gestisce l'accoppiamento e lo spawn di nuovi boid
    //spawnNewBoids();

    copyBoidsToGPU(boids, gpuBoids);

    //profiler.log("update", profiler.stop());
}

void SimulationGPU::render()
{
    profiler.start();

    // 1. Draw grid
    glLineWidth(1.0f);
    gridRenderer->draw(wallGrid, glm::vec3(0.2f, 0.2f, 0.2f));
    //gridRenderer->draw(boidGrid, glm::vec3(0.6f, 0.6f, 0.6f));

    // 2. Draw boids
    /*for (const Boid& b : boids) {
        float angle = glm::degrees(atan2(b.velocity.y, b.velocity.x)) + 270.0f;
        boidRenderer->draw(b.position, angle, b.color, 10.0f * b.scale);
    }*/

    std::vector<glm::vec2> positions;
    std::vector<float> rotations;
    std::vector<glm::vec3> colors;
    std::vector<float> scales;

    for (const auto& b : boids) {
        positions.push_back(b.position);
        rotations.push_back(glm::degrees(atan2(b.velocity.y, b.velocity.x)) + 270.0f);
        colors.push_back(b.color);
        scales.push_back(10.0f * b.scale);
    }

    boidRenderer->updateInstances(positions, rotations, colors, scales);
    boidRenderer->draw();

    // 3. Draw walls
    glLineWidth(3.0f);
    for (const Wall& w : walls)
        wallRenderer->draw(w, glm::vec3(0.25f, 0.88f, 0.82f));

    profiler.log("render", profiler.stop());

    // 4. Draw debug vectors
    //for (Boid& b : boids) {
    //    if (b.type == LEADER || b.type == PREDATOR) {
    //        glm::vec2 start = b.position;

    //        vectorRenderer->DrawVector(start, start + b.debugVectors[0] * 20.0f, glm::vec3(1.0f, 1.0f, 0.0f)); // giallo
    //        vectorRenderer->DrawVector(start, start + b.debugVectors[1] * 20.0f, glm::vec3(1.0f, 0.0f, 0.0f)); // rosso
    //        vectorRenderer->DrawVector(start, start + b.debugVectors[2] * 20.0f, glm::vec3(0.05f, 0.8f, 0.7f)); // ciano
    //        vectorRenderer->DrawVector(start, start + b.debugVectors[3] * 20.0f, glm::vec3(0.0f, 0.0f, 1.0f)); // blu
    //        //vectorRenderer->DrawVector(start, start + b.debugVectors[4] * 20.0f, glm::vec3(1.0f, 0.0f, 1.0f)); // magenta
    //    }
    //}

    // 5. Draw HUD / stats
    float margin = 10.0f;
    float scale = 0.7f;
    glm::vec3 color(0.9f, 0.9f, 0.3f);

    double fps = profiler.getCurrentFPS();
    if (fps > 0.0)
        textRenderer->draw("FPS: " + std::to_string(static_cast<int>(fps)), margin, height - margin - 20.0f, scale, color);

    textRenderer->draw("Boids: " + std::to_string(boids.size()), margin, height - margin - 40.0f, scale, color);
}

void SimulationGPU::processInput(float dt) {}

void SimulationGPU::updateStats(float dt) {
    profiler.updateFrameStats(dt);
}

void SimulationGPU::saveProfilerCSV(const std::string& path) {
    profiler.saveCSV(path);
}


// === HELPER Init ===
void SimulationGPU::initLeaders(int count)
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

void SimulationGPU::initPrey(int count)
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

void SimulationGPU::initPredators(int count)
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

void SimulationGPU::initWalls(int count)
{
    auto candidates = wallGrid.cellEdges;
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
void SimulationGPU::computeForces(std::vector<glm::vec2>& velocityChanges) {
    int N = static_cast<int>(boids.size());

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // --- Alloca temporaneamente velChange sul device ---
    float* d_velChangeX = nullptr;
    float* d_velChangeY = nullptr;
    cudaMalloc(&d_velChangeX, N * sizeof(float));
    cudaMalloc(&d_velChangeY, N * sizeof(float));

    // Inizializza a zero
    cudaMemset(d_velChangeX, 0, N * sizeof(float));
    cudaMemset(d_velChangeY, 0, N * sizeof(float));

    printf("computeForces: N=%d, cohesionDistance=%.3f, cohesionScale=%.3f, "
        "separationDistance=%.3f, separationScale=%.3f, "
        "alignmentDistance=%.3f, alignmentScale=%.3f\n",
        N,
        params.cohesionDistance, params.cohesionScale,
        params.separationDistance, params.separationScale,
        params.alignmentDistance, params.alignmentScale);

    // Lancia il kernel usando gli array temporanei
    computeForcesKernel << <blocks, threads >> > (
        N,
        gpuBoids.posX, gpuBoids.posY,
        gpuBoids.velX, gpuBoids.velY,
        gpuBoids.influence,
        gpuBoids.type,
        d_velChangeX,  // array temporaneo
        d_velChangeY,  // array temporaneo
        params.cohesionDistance, params.cohesionScale,
        params.separationDistance, params.separationScale,
        params.alignmentDistance, params.alignmentScale
    );

    cudaDeviceSynchronize();

    // Copia i risultati sul CPU
    velocityChanges.resize(N);
    std::vector<float> tmpX(N), tmpY(N);
    cudaMemcpy(tmpX.data(), d_velChangeX, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmpY.data(), d_velChangeY, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        velocityChanges[i] = glm::vec2(tmpX[i], tmpY[i]);

    // Libera la memoria temporanea
    cudaFree(d_velChangeX);
    cudaFree(d_velChangeY);
}

void SimulationGPU::applyVelocity(float dt, std::vector<glm::vec2>& velocityChanges)
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

void SimulationGPU::checkEatenPrey()
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

void SimulationGPU::spawnNewBoids()
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