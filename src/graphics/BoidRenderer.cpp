#include <graphics/BoidRenderer.h>

BoidRenderer::BoidRenderer(const Shader& shader)
    : shader(shader) {
    initBuffers();
}

BoidRenderer::~BoidRenderer() {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

void BoidRenderer::draw(glm::vec2 position, float rotation, glm::vec3 color, float scale) {
    shader.Use();

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(position, 0.0f));
    model = glm::rotate(model, glm::radians(rotation), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::scale(model, glm::vec3(scale, scale, 1.0f));

    shader.SetMatrix4("model", model);
    shader.SetVector3f("spriteColor", color);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void BoidRenderer::initBuffers() {
    // Triangolo centrato sull'origine
    float vertices[] = {
        0.0f,   0.5f,   // punta
       -0.25f, -0.25f,  // sinistra
        0.25f, -0.25f   // destra
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
