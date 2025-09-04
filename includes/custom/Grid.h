#pragma once
#ifndef GRID_H
#define GRID_H

#include <vector>
#include <glm/glm.hpp>

struct Grid {
    int nRows;
    int nCols;
    float width;
    float height;

    std::vector<std::pair<glm::vec2, glm::vec2>> lines;        // linee complete (da bordo a bordo)
    std::vector<glm::vec2> intersections;                      // tutti gli incroci della griglia
    std::vector<std::pair<glm::vec2, glm::vec2>> cellEdges;    // lati di cella (da incrocio a incrocio)

    Grid(int rows, int cols, float w, float h)
        : nRows(rows), nCols(cols), width(w), height(h)
    {
        generateLines();
        generateIntersections();
        generateCellEdges();
    }

    void generateLines() {
        lines.clear();
        float rowSpacing = height / (nRows + 1);
        float colSpacing = width / (nCols + 1);

        // linee orizzontali
        for (int i = 1; i <= nRows; i++) {
            float y = i * rowSpacing;
            lines.push_back({
                glm::vec2(0.0f, y),
                glm::vec2(width, y)
                });
        }

        // linee verticali
        for (int j = 1; j <= nCols; j++) {
            float x = j * colSpacing;
            lines.push_back({
                glm::vec2(x, 0.0f),
                glm::vec2(x, height)
                });
        }
    }

    void generateIntersections() {
        intersections.clear();
        float rowSpacing = height / (nRows + 1);
        float colSpacing = width / (nCols + 1);

        for (int i = 1; i <= nRows; i++) {
            float y = i * rowSpacing;
            for (int j = 1; j <= nCols; j++) {
                float x = j * colSpacing;
                intersections.push_back(glm::vec2(x, y));
            }
        }
    }

    void generateCellEdges() {
        cellEdges.clear();
        float rowSpacing = height / (nRows + 1);
        float colSpacing = width / (nCols + 1);

        for (int i = 1; i <= nRows; i++) {
            float y = i * rowSpacing;
            for (int j = 1; j <= nCols; j++) {
                float x = j * colSpacing;

                glm::vec2 current(x, y);

                // lato destro (se non è l’ultima colonna)
                if (j < nCols) {
                    glm::vec2 right(x + colSpacing, y);
                    cellEdges.push_back({ current, right });
                }

                // lato in basso (se non è l’ultima riga)
                if (i < nRows) {
                    glm::vec2 down(x, y + rowSpacing);
                    cellEdges.push_back({ current, down });
                }
            }
        }
    }
};

#endif
