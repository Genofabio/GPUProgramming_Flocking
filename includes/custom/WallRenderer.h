#pragma once
#ifndef WALL_RENDERER_H
#define WALL_RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include "Shader.h"
#include "Wall.h"

class WallRenderer {
public:
    WallRenderer(const Shader& shader);
    ~WallRenderer();

    // Disegna un muro tra due punti
    void DrawWall(const Wall& wall, glm::vec3 color);

private:
    Shader shader;
    unsigned int lineVAO, lineVBO;
};

#endif
