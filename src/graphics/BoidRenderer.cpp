#include <graphics/BoidRenderer.h>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>

BoidRenderer::BoidRenderer(const Shader& shader)
    : shader(shader)
{
    initBuffers();
}

BoidRenderer::~BoidRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &instanceVBO);
}

// aggiorniamo le posizioni dei boid ogni frame
void BoidRenderer::updateInstances(const std::vector<glm::vec2>& positions,
    const std::vector<float>& rotations,
    const std::vector<glm::vec3>& colors,
    const std::vector<float>& scales)
{
    size_t N = positions.size();
    instanceData.resize(N);

    for (size_t i = 0; i < N; ++i) {
        glm::mat4 model(1.0f);
        model = glm::translate(model, glm::vec3(positions[i], 0.0f));
        model = glm::rotate(model, glm::radians(rotations[i]), glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, glm::vec3(scales[i], scales[i], 1.0f));

        instanceData[i].model = model;
        instanceData[i].color = colors[i];
    }

    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(InstanceData), instanceData.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void BoidRenderer::draw() {
    shader.Use();
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLES, 0, 3, instanceData.size());
    glBindVertexArray(0);
}

void BoidRenderer::initBuffers() {
    // triangolo base centrato sull'origine
    float vertices[] = {
        0.0f, 0.5f,   // punta
       -0.25f, -0.25f,// sinistra
        0.25f, -0.25f // destra
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &instanceVBO);

    glBindVertexArray(vao);

    // vertici del triangolo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // buffer istanze (matrici + colori)
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

    // modello (mat4) occupa 4 attributi consecutivi
    for (int i = 0; i < 4; ++i) {
        glEnableVertexAttribArray(1 + i);
        glVertexAttribPointer(1 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)(i * sizeof(glm::vec4)));
        glVertexAttribDivisor(1 + i, 1); // una volta per istanza
    }

    // colore
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, color));
    glVertexAttribDivisor(5, 1);

    glBindVertexArray(0);
}
