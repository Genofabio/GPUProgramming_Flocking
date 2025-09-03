#ifndef BOID_RENDERER_H
#define BOID_RENDERER_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Shader.h"

class BoidRenderer {
public:
    BoidRenderer(const Shader& shader);
    ~BoidRenderer();

    // Disegna un singolo boid come triangolo
    void DrawBoid(glm::vec2 position, float rotation, glm::vec3 color, float scale = 10.0f);

private:
    Shader shader;
    unsigned int triangleVAO;

    void initRenderData();
};

#endif#pragma once
