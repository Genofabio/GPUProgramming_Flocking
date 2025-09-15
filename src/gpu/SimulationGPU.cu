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

#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define CUDA_CHECK(err) \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

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
    initPrey(3000);
    initPredators(0);

    // Allocate and copy boid data to GPU
    if (!boidDataInitialized) {
        allocateBoidDataGPU(gpuBoids, boids.size());
        copyBoidsToGPU(boids, gpuBoids);
        boidDataInitialized = true;
    }

    // Initialize walls
    initWalls(50);

    // Numero boid e celle griglia
    size_t N = boids.size();
    size_t numCells = boidGrid.nRows * boidGrid.nCols;

    // Allocazione dei buffer della uniform grid
    allocateGridBuffers(N, numCells);
}

void SimulationGPU::update(float dt) {
    profiler.start();

    profiler.start();
    currentTime += dt;
    size_t N = boids.size();
    if (N == 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 1. Calcola le forze
    computeForces();

    // 2. Applica le variazioni di velocità dai buffer sorted
    kernApplyVelocityChangeSorted << <blocks, threads >> > (
        static_cast<int>(N),
        gpuBoids.velChangeX_sorted, gpuBoids.velChangeY_sorted,
        gpuBoids.posX, gpuBoids.posY,
        gpuBoids.velX, gpuBoids.velY,
        dev_particleArrayIndices,
        dt, params.slowDownFactor, params.maxSpeed
        );
    cudaDeviceSynchronize();

    // 3. Calcola le rotazioni dai vettori velocità (rimane sui buffer originali)
    kernComputeRotations << <blocks, threads >> > (
        static_cast<int>(N),
        gpuBoids.velX, gpuBoids.velY,
        gpuBoids.rotations
        );
    cudaDeviceSynchronize();

    // --- 4. Copia solo i dati necessari per il rendering ---
    renderPositions.resize(N);
    renderRotations.resize(N);
    renderColors.resize(N);
    renderScales.resize(N);

    std::vector<float> posX(N), posY(N);
    CUDA_CHECK(cudaMemcpy(posX.data(), gpuBoids.posX, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(posY.data(), gpuBoids.posY, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; i++) {
        renderPositions[i] = { posX[i], posY[i] };
    }

    CUDA_CHECK(cudaMemcpy(renderRotations.data(), gpuBoids.rotations, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> colorR(N), colorG(N), colorB(N);
    CUDA_CHECK(cudaMemcpy(colorR.data(), gpuBoids.colorR, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colorG.data(), gpuBoids.colorG, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(colorB.data(), gpuBoids.colorB, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; i++) {
        renderColors[i] = { colorR[i], colorG[i], colorB[i] };
    }

    CUDA_CHECK(cudaMemcpy(renderScales.data(), gpuBoids.scale, N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- 5. Resetta i delta velocità ---
    // 5. Resetta i delta velocità sorted
    cudaMemset(gpuBoids.velChangeX_sorted, 0, N * sizeof(float));
    cudaMemset(gpuBoids.velChangeY_sorted, 0, N * sizeof(float));

    // --- DEBUG: stampa primi boid ---
    /*
    for (int i = 0; i < std::min<size_t>(N, 5); i++) {
        std::cout << "Boid " << i
                  << " pos=(" << renderPositions[i].x << ", " << renderPositions[i].y << ")"
                  << " vel=(";
        float vx, vy;
        CUDA_CHECK(cudaMemcpy(&vx, gpuBoids.velX + i, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&vy, gpuBoids.velY + i, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << vx << ", " << vy << ")"
                  << " rot=" << renderRotations[i]
                  << " scale=" << renderScales[i]
                  << " color=(" << renderColors[i].r << "," << renderColors[i].g << "," << renderColors[i].b << ")"
                  << std::endl;
    }
    */

    // --- 6. Aggiorna il renderer ---
    for (size_t i = 0; i < N; i++) {
        renderScales[i] *= 8.0f;
    }
    boidRenderer->updateInstances(renderPositions, renderRotations, renderColors, renderScales);

    profiler.log("update", profiler.stop());
}


void SimulationGPU::render()
{
    profiler.start();

    // 1. Draw grid
    glLineWidth(1.0f);
    gridRenderer->draw(wallGrid, glm::vec3(0.2f, 0.2f, 0.2f));
    //gridRenderer->draw(boidGrid, glm::vec3(0.6f, 0.6f, 0.6f));

    // 2. Draw boids (usa i vettori aggiornati da update())
    boidRenderer->draw();

    // 3. Draw walls
    glLineWidth(3.0f);
    for (const Wall& w : walls)
        wallRenderer->draw(w, glm::vec3(0.25f, 0.88f, 0.82f));

    profiler.log("render", profiler.stop());

    // 4. Draw debug vectors
    // Qui potremmo in futuro leggere i vettori di debug dalla GPU
    // per ora resta commentato o continua a usare boids CPU se servono test

    // 5. Draw HUD / stats
    float margin = 10.0f;
    float scale = 0.7f;
    glm::vec3 color(0.9f, 0.9f, 0.3f);

    double fps = profiler.getCurrentFPS();
    if (fps > 0.0)
        textRenderer->draw("FPS: " + std::to_string(static_cast<int>(fps)),
            margin, height - margin - 20.0f, scale, color);

    textRenderer->draw("Boids: " + std::to_string(boids.size()),
        margin, height - margin - 40.0f, scale, color);
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
void SimulationGPU::computeForces() {
    int N = static_cast<int>(boids.size());
    if (N == 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 1. Calcola gli indici della griglia
    kernComputeIndices << <blocks, threads >> > (
        N,
        gpuBoids.posX, gpuBoids.posY,
        dev_particleGridIndices,
        dev_particleArrayIndices,
        boidGrid.nCols, boidGrid.nRows,
        0.0f, 0.0f,
        boidGrid.cellWidth
        );
    cudaDeviceSynchronize();

    // 2. Ordina per cella
    thrust::device_ptr<int> devGridKeys(dev_particleGridIndices);
    thrust::device_ptr<int> devArrayIndices(dev_particleArrayIndices);
    thrust::sort_by_key(devGridKeys, devGridKeys + N, devArrayIndices);

    // 3. Reorder dei buffer
    kernReorderData << <blocks, threads >> > (
        N,
        gpuBoids.posX, gpuBoids.posY,
        gpuBoids.velX, gpuBoids.velY,
        gpuBoids.scale, gpuBoids.influence,
        gpuBoids.type,
        gpuBoids.colorR, gpuBoids.colorG, gpuBoids.colorB,
        gpuBoids.velChangeX, gpuBoids.velChangeY,
        dev_particleArrayIndices,
        gpuBoids.posX_sorted, gpuBoids.posY_sorted,
        gpuBoids.velX_sorted, gpuBoids.velY_sorted,
        gpuBoids.scale_sorted, gpuBoids.influence_sorted,
        gpuBoids.type_sorted,
        gpuBoids.colorR_sorted, gpuBoids.colorG_sorted, gpuBoids.colorB_sorted,
        gpuBoids.velChangeX_sorted, gpuBoids.velChangeY_sorted
        );
    cudaDeviceSynchronize();

    // 4. Trova start/end per ogni cella
    kernIdentifyCellStartEnd << <blocks, threads >> > (
        N,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices
        );
    cudaDeviceSynchronize();

    // 5. Calcola le forze sui buffer ordinati
    computeForcesKernelGridOptimized << <blocks, threads >> > (
        N,
        gpuBoids.posX_sorted, gpuBoids.posY_sorted,
        gpuBoids.velX_sorted, gpuBoids.velY_sorted,
        gpuBoids.influence_sorted,
        gpuBoids.type_sorted,
        dev_particleArrayIndices,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        boidGrid.nCols, boidGrid.nRows,
        boidGrid.cellWidth,
        params.cohesionDistance, params.cohesionScale,
        params.separationDistance, params.separationScale,
        params.alignmentDistance, params.alignmentScale,
        static_cast<float>(width),
        static_cast<float>(height),
        params.borderAlertDistance,
        gpuBoids.velChangeX_sorted,
        gpuBoids.velChangeY_sorted
        );
    cudaDeviceSynchronize();
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

void SimulationGPU::allocateGridBuffers(size_t N, size_t numCells)
{
    // Prima liberiamo eventuali buffer già allocati
    freeGridBuffers();

    // Allochiamo i buffer legati ai boid
    if (cudaMalloc(&dev_particleGridIndices, N * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: dev_particleGridIndices");

    if (cudaMalloc(&dev_particleArrayIndices, N * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: dev_particleArrayIndices");

    // Allochiamo i buffer legati alle celle della griglia
    if (cudaMalloc(&dev_gridCellStartIndices, numCells * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: dev_gridCellStartIndices");

    if (cudaMalloc(&dev_gridCellEndIndices, numCells * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed: dev_gridCellEndIndices");

    // Inizializziamo le celle a -1 (vuote)
    cudaMemset(dev_gridCellStartIndices, -1, numCells * sizeof(int));
    cudaMemset(dev_gridCellEndIndices, -1, numCells * sizeof(int));
}

void SimulationGPU::freeGridBuffers()
{
    if (dev_particleGridIndices) {
        cudaFree(dev_particleGridIndices);
        dev_particleGridIndices = nullptr;
    }
    if (dev_particleArrayIndices) {
        cudaFree(dev_particleArrayIndices);
        dev_particleArrayIndices = nullptr;
    }
    if (dev_gridCellStartIndices) {
        cudaFree(dev_gridCellStartIndices);
        dev_gridCellStartIndices = nullptr;
    }
    if (dev_gridCellEndIndices) {
        cudaFree(dev_gridCellEndIndices);
        dev_gridCellEndIndices = nullptr;
    }
}
