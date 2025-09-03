#include "custom/BoidRenderer.h"

BoidRenderer::BoidRenderer(const Shader& shader)
{
    this->shader = shader;
    this->initRenderData();
}

BoidRenderer::~BoidRenderer()
{
    glDeleteVertexArrays(1, &this->triangleVAO);
}

void BoidRenderer::DrawBoid(glm::vec2 position, float rotation, glm::vec3 color, float scale)
{
    this->shader.Use();

    glm::mat4 model = glm::mat4(1.0f);

    // Traslazione alla posizione
    model = glm::translate(model, glm::vec3(position, 0.0f));

    // Rotazione attorno all'origine (triangolo già centrato)
    model = glm::rotate(model, glm::radians(rotation), glm::vec3(0.0f, 0.0f, 1.0f));

    // Scala uniforme
    model = glm::scale(model, glm::vec3(scale, scale, 1.0f));

    this->shader.SetMatrix4("model", model);
    this->shader.SetVector3f("spriteColor", color);

    glBindVertexArray(this->triangleVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void BoidRenderer::initRenderData()
{
    // Triangolo centrato sull'origine (centroide in 0,0)
    float vertices[] = {
        0.0f,  0.5f,    // punta più alta
       -0.25f, -0.25f,  // lato sinistro più stretto
        0.25f, -0.25f   // lato destro più stretto
    };

    unsigned int VBO;
    glGenVertexArrays(1, &this->triangleVAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(this->triangleVAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
