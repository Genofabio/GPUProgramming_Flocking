#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>
#include "../core/Boid.h"

struct BoidData {
    size_t N = 0;
    float* posX = nullptr;
    float* posY = nullptr;
    float* velX = nullptr;
    float* velY = nullptr;
    float* driftX = nullptr;
    float* driftY = nullptr;
    float* scale = nullptr;
    float* influence = nullptr;
    int* type = nullptr;
    int* age = nullptr;
    float* birthTime = nullptr;
    float* velChangeX = nullptr;
    float* velChangeY = nullptr;
    float* rotations = nullptr;
    float* colorR = { nullptr };
    float* colorG = { nullptr };
    float* colorB = { nullptr };
    float* debugX[5] = { nullptr };
    float* debugY[5] = { nullptr };
};

void allocateBoidDataGPU(BoidData& bd, size_t N);
void copyBoidsToGPU(const std::vector<Boid>& cpuBoids, BoidData& bd);
void copyBoidsToCPU(BoidData& bd, std::vector<Boid>& cpuBoids);
void freeBoidDataGPU(BoidData& bd);