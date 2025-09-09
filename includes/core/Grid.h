#pragma once
#include <vector>
#include <external/glm/glm.hpp>

struct Grid {
    int nRows;
    int nCols;
    float width;
    float height;

    std::vector<std::pair<glm::vec2, glm::vec2>> lines;
    std::vector<glm::vec2> intersections;
    std::vector<std::pair<glm::vec2, glm::vec2>> cellEdges;

    Grid(int rows, int cols, float w, float h)
        : nRows(rows), nCols(cols), width(w), height(h)
    {
        float rowSpacing = height / (nRows + 1);
        float colSpacing = width / (nCols + 1);

        generateLines(rowSpacing, colSpacing);
        generateIntersections(rowSpacing, colSpacing);
        generateCellEdges(rowSpacing, colSpacing);
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