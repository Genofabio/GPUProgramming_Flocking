#pragma once
#include <cstddef>
#include <cuda_runtime.h>

struct GridData {
    int* particleGridIndices = nullptr;
    int* particleArrayIndices = nullptr;
    int* cellStartIndices = nullptr;
    int* cellEndIndices = nullptr;
    size_t numCells = 0;
};

void allocateGridDataGPU(GridData& gd, size_t N, size_t numCells);
void freeGridDataGPU(GridData& gd);
