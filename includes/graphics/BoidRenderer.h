#pragma once
#ifndef BOID_RENDERER_H
#define BOID_RENDERER_H

#include <external/glad/glad.h>
#include <external/glm/glm.hpp>
#include <external/glm/gtc/matrix_transform.hpp>
#include <vector>
#include <graphics/Shader.h>

class BoidRenderer {
public:
    explicit BoidRenderer(const Shader& shader);
    ~BoidRenderer();

    // Aggiorna le istanze dei boid (posizioni, rotazioni, colori, scale)
    void updateInstances(const std::vector<glm::vec2>& positions,
        const std::vector<float>& rotations,
        const std::vector<glm::vec3>& colors,
        const std::vector<float>& scales);

    // Disegna tutti i boid aggiornati
    void draw();

private:
    Shader shader;

    unsigned int vao = 0;
    unsigned int vbo = 0;
    unsigned int instanceVBO = 0;

    struct InstanceData {
        glm::mat4 model;
        glm::vec3 color;
    };

    std::vector<InstanceData> instanceData;

    void initBuffers();
};

#endif
