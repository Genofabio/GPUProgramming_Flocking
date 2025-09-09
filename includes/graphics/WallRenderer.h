#pragma once
#ifndef WALL_RENDERER_H
#define WALL_RENDERER_H

#include <external/glad/glad.h>
#include <external/glm/glm.hpp>
#include <graphics/Shader.h>
#include <core/Wall.h>

class WallRenderer {
public:
    explicit WallRenderer(const Shader& shader);
    ~WallRenderer();

    void draw(const Wall& wall, const glm::vec3& color);

private:
    Shader shader;
    unsigned int vao = 0;
    unsigned int vbo = 0;

    void initBuffers();
};

#endif