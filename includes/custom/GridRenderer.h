#pragma once
#ifndef GRID_RENDERER_H
#define GRID_RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Grid.h"

class GridRenderer {
public:
    GridRenderer(const Shader& shader);
    ~GridRenderer();

    // Disegna la griglia 
    void DrawGrid(const Grid& grid, glm::vec3 color);

private:
    Shader shader;
    unsigned int lineVAO, lineVBO;
};

#endif
