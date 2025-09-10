#pragma once
#include <vector>
#include <external/glm/glm.hpp>

struct GridCell {
    std::vector<size_t> boidIndices; // Indici dei boid presenti nella cella
};

struct Grid {
    int nRows;
    int nCols;
    float width;
    float height;
    float cellWidth;
    float cellHeight;

    std::vector<std::pair<glm::vec2, glm::vec2>> lines;
    std::vector<glm::vec2> intersections;
    std::vector<std::pair<glm::vec2, glm::vec2>> cellEdges;

    // Nuovo: celle per la uniform grid dei boid
    std::vector<GridCell> cells;

    Grid(int rows, int cols, float w, float h)
        : nRows(rows), nCols(cols), width(w), height(h)
    {
        cellWidth = width / nCols;
        cellHeight = height / nRows;

        // Inizializza le celle vuote
        cells.resize(nRows * nCols);

        float rowSpacing = height / (nRows + 1);
        float colSpacing = width / (nCols + 1);

        generateLines(rowSpacing, colSpacing);
        generateIntersections(rowSpacing, colSpacing);
        generateCellEdges(rowSpacing, colSpacing);
    }

    // Aggiorna le celle con i boid attuali
    void updateCells(const std::vector<glm::vec2>& boidPositions) {
        // Pulisce tutte le celle
        for (auto& cell : cells)
            cell.boidIndices.clear();

        // Inserisce i boid nelle celle corrispondenti
        for (size_t i = 0; i < boidPositions.size(); ++i) {
            int col = static_cast<int>(boidPositions[i].x / cellWidth);
            int row = static_cast<int>(boidPositions[i].y / cellHeight);

            // Clamp per evitare out-of-bounds
            if (col < 0) col = 0;
            if (col >= nCols) col = nCols - 1;
            if (row < 0) row = 0;
            if (row >= nRows) row = nRows - 1;

            cells[row * nCols + col].boidIndices.push_back(i);
        }
    }

    // Restituisce gli indici dei boid nelle celle vicine alla cella (row, col)
    std::vector<size_t> getNearbyBoids(int row, int col) const {
        std::vector<size_t> nearby;

        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                int r = row + dr;
                int c = col + dc;
                if (r < 0 || r >= nRows || c < 0 || c >= nCols)
                    continue;
                const GridCell& cell = cells[r * nCols + c];
                nearby.insert(nearby.end(), cell.boidIndices.begin(), cell.boidIndices.end());
            }
        }
        return nearby;
    }

private:
    void generateLines(float rowSpacing, float colSpacing) {
        lines.clear();
        for (int i = 1; i <= nRows; i++)
            lines.push_back({ glm::vec2(0.0f, i * rowSpacing), glm::vec2(width, i * rowSpacing) });
        for (int j = 1; j <= nCols; j++)
            lines.push_back({ glm::vec2(j * colSpacing, 0.0f), glm::vec2(j * colSpacing, height) });
    }

    void generateIntersections(float rowSpacing, float colSpacing) {
        intersections.clear();
        for (int i = 1; i <= nRows; i++)
            for (int j = 1; j <= nCols; j++)
                intersections.push_back(glm::vec2(j * colSpacing, i * rowSpacing));
    }

    void generateCellEdges(float rowSpacing, float colSpacing) {
        cellEdges.clear();
        for (int i = 1; i <= nRows; i++) {
            for (int j = 1; j <= nCols; j++) {
                glm::vec2 current(j * colSpacing, i * rowSpacing);
                if (j < nCols) cellEdges.push_back({ current, glm::vec2((j + 1) * colSpacing, i * rowSpacing) });
                if (i < nRows) cellEdges.push_back({ current, glm::vec2(j * colSpacing, (i + 1) * rowSpacing) });
            }
        }
    }
};
