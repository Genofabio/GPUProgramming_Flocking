#include <set>
#include <iostream>
#include <cmath>
#include <random>

#include <gpu/SimulationGPU.cuh>
#include <utility/ResourceManager.h>
#include <gpu/BoidData.h>
#include <core/Boid.h>
#include <gpu/CudaKernels.cuh>
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
    params.maxSpeed = 100.0f;
    params.slowDownFactor = 0.3f;

    // Distanze 
    params.cohesionDistance = 60.0f;
    params.separationDistance = 20.0f;
    params.alignmentDistance = 40.0f;
    params.borderDistance = 80.0f;
    params.predatorFearDistance = 100.0f;
    params.predatorChaseDistance = 120.0f;
    params.predatorSeparationDistance = 50.0f;
    params.predatorEatDistance = 5.0f;

    // Scale (forza delle regole)
    params.cohesionScale = 0.1f;
    params.separationScale = 2.2f;
    params.alignmentScale = 0.19f;
    params.borderScale = 0.3f;
    params.predatorFearScale = 0.8f;
    params.predatorChaseScale = 0.12f;
    params.predatorSeparationScale = 2.0f;
    params.borderAlertDistance = 120.0f;

    // Muri
    params.wallRepulsionDistance = 50.0f; 
    params.wallRepulsionScale = 5.0f;     

    // Social/extra
    params.leaderInfluenceDistance = 120.0f;
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

    if (devRenderPositions) { cudaFree(devRenderPositions); devRenderPositions = nullptr; }
    if (devRenderRotations) { cudaFree(devRenderRotations); devRenderRotations = nullptr; }
    if (devRenderColors) { cudaFree(devRenderColors); devRenderColors = nullptr; }
    if (devRenderScales) { cudaFree(devRenderScales); devRenderScales = nullptr; }

    if (wallsDevicePositions) cudaFree(wallsDevicePositions);

    freeGridDataGPU(gridData);
    freeBoidDataGPU(gpuBoids);
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
    initPrey(5000);
    initPredators(0);

    // Allocate and copy boid data to GPU if not done yet
    if (!boidDataInitialized) {
        allocateBoidDataGPU(gpuBoids, boids.size());
        copyBoidsToGPU(boids, gpuBoids);
        boidDataInitialized = true;
    }

    // Initialize walls
    initWalls(50);

	// Preprocessing wall data for GPU
    prepareWallsGPU();

    // Allocate grid buffers
    size_t N = boids.size();
    size_t numCells = boidGrid.nRows * boidGrid.nCols;
    allocateGridDataGPU(gridData, N, numCells);

    // Allocate rendering buffers
    CUDA_CHECK(cudaMalloc(&devRenderPositions, N * sizeof(glm::vec2)));
    CUDA_CHECK(cudaMalloc(&devRenderRotations, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devRenderColors, N * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMalloc(&devRenderScales, N * sizeof(float)));
}

void SimulationGPU::update(float dt) {
    profiler.start();
    currentTime += dt;
    size_t N = boids.size();
    if (N == 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // --- 1. Resetta i delta velocità ---
    cudaMemset(gpuBoids.velChangeX_sorted, 0, N * sizeof(float));
    cudaMemset(gpuBoids.velChangeY_sorted, 0, N * sizeof(float));

    // --- 2. Calcola le forze ---
    computeForces();

    // --- 3. Applica le variazioni di velocità dai buffer sorted ---
    kernApplyVelocityChangeSorted << <blocks, threads >> > (
        static_cast<int>(N),
        gpuBoids.velChangeX_sorted, gpuBoids.velChangeY_sorted,
        gpuBoids.posX, gpuBoids.posY,
        gpuBoids.velX, gpuBoids.velY,
        gridData.particleArrayIndices, // <--- sostituito
        dt, params.slowDownFactor, params.maxSpeed
        );

    // --- 4. Calcola le rotazioni dai vettori velocità ---
    kernComputeRotations << <blocks, threads >> > (
        static_cast<int>(N),
        gpuBoids.velX, gpuBoids.velY,
        gpuBoids.rotations
        );

    // --- 5. Copia i dati per il rendering sui buffer device ---
    copyRenderDataKernel << <blocks, threads >> > (
        static_cast<int>(N),
        gpuBoids.posX, gpuBoids.posY,
        gpuBoids.rotations,
        gpuBoids.colorR, gpuBoids.colorG, gpuBoids.colorB,
        gpuBoids.scale,
        devRenderPositions,
        devRenderRotations,
        devRenderColors,
        devRenderScales
        );

    // --- 6. Sincronizza prima di copiare i dati su CPU ---
    cudaDeviceSynchronize();

    // --- 7. Copia i dati da GPU a CPU ---
    renderPositions.resize(N);
    renderRotations.resize(N);
    renderColors.resize(N);
    renderScales.resize(N);

    CUDA_CHECK(cudaMemcpy(renderPositions.data(), devRenderPositions, N * sizeof(glm::vec2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(renderRotations.data(), devRenderRotations, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(renderColors.data(), devRenderColors, N * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(renderScales.data(), devRenderScales, N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- 8. Aggiorna il renderer ---
    boidRenderer->updateInstances(renderPositions, renderRotations, renderColors, renderScales);

    profiler.log("update", profiler.stop());
}

void SimulationGPU::render()
{
    profiler.start();

    // Grid
    glLineWidth(1.0f);
    gridRenderer->draw(wallGrid, glm::vec3(0.2f, 0.2f, 0.2f));

    // Boids
    boidRenderer->draw();

    // Walls
    glLineWidth(3.0f);
    for (const Wall& w : walls)
        wallRenderer->draw(w, glm::vec3(0.25f, 0.88f, 0.82f));

    profiler.log("render", profiler.stop());

    // HUD / Stats
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
        b.scale = (1.0f + 0.04f * b.age) * 8.0f;
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
        walls.push_back(w);  

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

void SimulationGPU::prepareWallsGPU() {
    // Conta i segmenti totali
    numWallSegments = 0;
    for (const auto& w : walls) {
        if (w.points.size() >= 2)
            numWallSegments += static_cast<size_t>(w.points.size() - 1);
    }

    if (numWallSegments == 0) return;

    // Alloca array host temporaneo: 2 punti per segmento
    std::vector<float2> hPositions(numWallSegments * 2);

    size_t idx = 0;
    for (const auto& w : walls) {
        for (size_t i = 0; i + 1 < w.points.size(); ++i) {
            glm::vec2 a = w.points[i];
            glm::vec2 b = w.points[i + 1];

            // Salva i due estremi del segmento
            hPositions[2 * idx] = make_float2(a.x, a.y);
            hPositions[2 * idx + 1] = make_float2(b.x, b.y);

            idx++;
        }
    }

    // Alloca GPU 
    CUDA_CHECK(cudaMalloc(&wallsDevicePositions, hPositions.size() * sizeof(float2)));

    // Copia dati su GPU
    CUDA_CHECK(cudaMemcpy(wallsDevicePositions, hPositions.data(),
        hPositions.size() * sizeof(float2), cudaMemcpyHostToDevice));
}

// === HELPER Update ===
void SimulationGPU::computeForces() {
    int N = static_cast<int>(boids.size());
    if (N == 0) return;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // --- 1. Calcola gli indici della griglia ---
    kernComputeIndices << <blocks, threads >> > (
        N,
        gpuBoids.posX, gpuBoids.posY,
        gridData.particleGridIndices,
        gridData.particleArrayIndices,
        boidGrid.nCols, boidGrid.nRows,
        0.0f, 0.0f,
        boidGrid.cellWidth
        );

    // --- 2. Ordina per cella ---
    thrust::device_ptr<int> devGridKeys(gridData.particleGridIndices);
    thrust::device_ptr<int> devArrayIndices(gridData.particleArrayIndices);
    thrust::sort_by_key(devGridKeys, devGridKeys + N, devArrayIndices);

    // --- 3. Reorder dei buffer ---
    kernReorderData << <blocks, threads >> > (
        N,
        gpuBoids.posX, gpuBoids.posY,
        gpuBoids.velX, gpuBoids.velY,
        gpuBoids.scale, gpuBoids.influence,
        gpuBoids.type,
        gpuBoids.colorR, gpuBoids.colorG, gpuBoids.colorB,
        gpuBoids.velChangeX, gpuBoids.velChangeY,
        gridData.particleArrayIndices,
        gpuBoids.posX_sorted, gpuBoids.posY_sorted,
        gpuBoids.velX_sorted, gpuBoids.velY_sorted,
        gpuBoids.scale_sorted, gpuBoids.influence_sorted,
        gpuBoids.type_sorted,
        gpuBoids.colorR_sorted, gpuBoids.colorG_sorted, gpuBoids.colorB_sorted,
        gpuBoids.velChangeX_sorted, gpuBoids.velChangeY_sorted
        );

    // --- 4. Trova start/end per ogni cella ---
    kernIdentifyCellStartEnd << <blocks, threads >> > (
        N,
        gridData.particleGridIndices,
        gridData.cellStartIndices,
        gridData.cellEndIndices
        );

    // --- 5. Calcola le forze sui buffer ordinati usando il kernel ottimizzato con tiling ---
    // Ogni boid necessita di 5 float in shared memory (posX, posY, velX, velY, influence)
	profiler.start();
    size_t shMemSize = threads * 5 * sizeof(float);

    computeForcesKernelAggressive << <blocks, threads, shMemSize >> > (
        N,
        gpuBoids.posX_sorted, gpuBoids.posY_sorted,
        gpuBoids.velX_sorted, gpuBoids.velY_sorted,
        gpuBoids.influence_sorted,
        gridData.cellStartIndices,
        gridData.cellEndIndices,
        boidGrid.nCols, boidGrid.nRows,
        boidGrid.cellWidth,
        params.cohesionDistance, params.cohesionScale,
        params.separationDistance, params.separationScale,
        params.alignmentDistance, params.alignmentScale,
        static_cast<float>(width),
        static_cast<float>(height),
        params.borderAlertDistance,
        gpuBoids.velChangeX_sorted,
        gpuBoids.velChangeY_sorted,
        numWallSegments,
        reinterpret_cast<float2*>(wallsDevicePositions),   
        params.wallRepulsionDistance,
        params.wallRepulsionScale
        );

    cudaDeviceSynchronize();
	profiler.log("compute forces", profiler.stop());
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
