#pragma once
#include <vector>
#include <external/glm/glm.hpp>

struct Grid {
    int nRows, nCols;
    float width, height;
    float cellWidth, cellHeight;
    int spacingCells; // padding in celle

    std::vector<std::pair<glm::vec2, glm::vec2>> lines;
    std::vector<glm::vec2> intersections;
    std::vector<std::pair<glm::vec2, glm::vec2>> cellEdges;

    Grid(int rows, int cols, float w, float h, int spacingCells_ = 0)
        : nRows(rows), nCols(cols), width(w), height(h), spacingCells(spacingCells_)
    {
        cellWidth = width / nCols;
        cellHeight = height / nRows;

        int usableRows = nRows - 2 * spacingCells;
        int usableCols = nCols - 2 * spacingCells;

        generateLines();           // tutte le linee, comprese quelle del padding
        generateIntersections(usableRows, usableCols);  // solo intersezioni interne
        generateCellEdges(usableRows, usableCols);      // solo celle interne
    }

private:
    void generateLines() {
        lines.clear();
        // linee orizzontali
        for (int i = 1; i < nRows; i++) {
            float y = i * cellHeight;
            lines.push_back({ {0.0f, y}, {width, y} });
        }
        // linee verticali
        for (int j = 1; j < nCols; j++) {
            float x = j * cellWidth;
            lines.push_back({ {x, 0.0f}, {x, height} });
        }
    }

    void generateIntersections(int usableRows, int usableCols) {
        intersections.clear();
        for (int i = 1; i < usableRows; i++) {
            for (int j = 1; j < usableCols; j++) {
                float x = (j + spacingCells) * cellWidth;
                float y = (i + spacingCells) * cellHeight;
                intersections.push_back({ x, y });
            }
        }
    }

    void generateCellEdges(int usableRows, int usableCols) {
        cellEdges.clear();
        for (int i = 0; i < usableRows; i++) {
            for (int j = 0; j < usableCols; j++) {
                float x0 = (j + spacingCells) * cellWidth;
                float y0 = (i + spacingCells) * cellHeight;
                float x1 = (j + 1 + spacingCells) * cellWidth;
                float y1 = (i + 1 + spacingCells) * cellHeight;

                cellEdges.push_back({ {x0,y0}, {x1,y0} });
                cellEdges.push_back({ {x1,y0}, {x1,y1} });
                cellEdges.push_back({ {x1,y1}, {x0,y1} });
                cellEdges.push_back({ {x0,y1}, {x0,y0} });
            }
        }
    }
};
