#pragma once
#ifndef GRID_RENDERER_H
#define GRID_RENDERER_H

#include <external/glad/glad.h>
#include <external/glm/glm.hpp>
#include <graphics/Shader.h>
#include <core/Grid.h>

class GridRenderer {
public:
    explicit GridRenderer(const Shader& shader);
    ~GridRenderer();

    void draw(const Grid& grid, const glm::vec3& color);

private:
    Shader shader;
    unsigned int vao = 0;
    unsigned int vbo = 0;

    void initBuffers();
};

#endif
