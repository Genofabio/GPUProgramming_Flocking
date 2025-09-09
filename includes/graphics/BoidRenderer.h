#pragma once
#ifndef BOID_RENDERER_H
#define BOID_RENDERER_H

#include <external/glad/glad.h>
#include <external/glm/glm.hpp>
#include <external/glm/gtc/matrix_transform.hpp>

#include <graphics/Shader.h>

class BoidRenderer {
public:
    explicit BoidRenderer(const Shader& shader);
    ~BoidRenderer();

    // Disegna un singolo boid come triangolo
    void draw(glm::vec2 position, float rotation, glm::vec3 color, float scale = 10.0f);

private:
    Shader shader;
    unsigned int vao = 0;
    unsigned int vbo = 0;

    void initBuffers();
};

#endif