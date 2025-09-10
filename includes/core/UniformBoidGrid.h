#pragma once
#include <core/Grid.h>
#include <vector>
#include <glm/glm.hpp>

struct GridCell {
    std::vector<size_t> boidIndices; // indici dei boid in quella cella
};

class UniformBoidGrid : public Grid {
public:
    std::vector<GridCell> cells; // celle contenenti i boid

    // Costruttore di default
    UniformBoidGrid()
        : Grid(0, 0, 0.0f, 0.0f)
    {
    }

    UniformBoidGrid(int rows, int cols, float w, float h)
        : Grid(rows, cols, w, h)
    {
        cells.resize(nRows * nCols);
    }

    // Aggiorna le celle con le posizioni correnti dei boid
    void updateCells(const std::vector<glm::vec2>& boidPositions) {
        for (auto& cell : cells)
            cell.boidIndices.clear();

        for (size_t i = 0; i < boidPositions.size(); ++i) {
            int col = static_cast<int>(boidPositions[i].x / cellWidth);
            int row = static_cast<int>(boidPositions[i].y / cellHeight);

            // clamp per sicurezza
            col = glm::clamp(col, 0, nCols - 1);
            row = glm::clamp(row, 0, nRows - 1);

            cells[row * nCols + col].boidIndices.push_back(i);
        }
    }

    // Restituisce tutti i boid nelle celle vicine
    std::vector<size_t> getNearbyBoids(const glm::vec2& boidPos) const {
        std::vector<size_t> nearby;

        int col = static_cast<int>(boidPos.x / cellWidth);
        int row = static_cast<int>(boidPos.y / cellHeight);

        col = glm::clamp(col, 0, nCols - 1);
        row = glm::clamp(row, 0, nRows - 1);

        float localX = (boidPos.x - col * cellWidth) / cellWidth;
        float localY = (boidPos.y - row * cellHeight) / cellHeight;

        int worldDx = (localX > 0.5f) ? 1 : -1;
        int worldDy = (localY > 0.5f) ? 1 : -1;

        const bool rowIncreasesWithY = true;

        int colOffset = worldDx;
        int rowOffset = rowIncreasesWithY ? worldDy : -worldDy;

        int dr[4] = { 0, rowOffset, 0, rowOffset };
        int dc[4] = { 0, 0, colOffset, colOffset };

        for (int i = 0; i < 4; ++i) {
            int r = row + dr[i];
            int c = col + dc[i];
            if (r < 0 || r >= nRows || c < 0 || c >= nCols) continue;

            const GridCell& cell = cells[r * nCols + c];
            nearby.insert(nearby.end(), cell.boidIndices.begin(), cell.boidIndices.end());
        }

        return nearby;
    }

};
