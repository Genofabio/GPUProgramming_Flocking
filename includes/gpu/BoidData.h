#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>
#include "../core/Boid.h"

struct BoidData {
    size_t N = 0;

    // --- Originali  ---
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
    float* colorR = nullptr;
    float* colorG = nullptr;
    float* colorB = nullptr;
    float* debugX[5] = { nullptr };
    float* debugY[5] = { nullptr };

    // --- Ordinati ---
    float* posX_sorted = nullptr;
    float* posY_sorted = nullptr;
    float* velX_sorted = nullptr;
    float* velY_sorted = nullptr;
    float* velChangeX_sorted = nullptr;
    float* velChangeY_sorted = nullptr;
    float* scale_sorted = nullptr;
    float* influence_sorted = nullptr;
    int* type_sorted = nullptr;
    float* colorR_sorted = nullptr;
    float* colorG_sorted = nullptr;
    float* colorB_sorted = nullptr;
};

void allocateBoidDataGPU(BoidData& bd, size_t N);
void copyBoidsToGPU(const std::vector<Boid>& cpuBoids, BoidData& bd);
void freeBoidDataGPU(BoidData& bd);