#pragma once
#ifndef TEXT_RENDERER_H
#define TEXT_RENDERER_H

#include <map>
#include <string>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <external/glad/glad.h>
#include <external/glm/glm.hpp>
#include <graphics/Shader.h>

struct Character {
    unsigned int TextureID;
    glm::ivec2   Size;
    glm::ivec2   Bearing;
    unsigned int Advance;
};

class TextRenderer {
public:
    explicit TextRenderer(const Shader& shader);
    ~TextRenderer();

    // Carica un font
    void loadFont(const std::string& fontPath, unsigned int fontSize);

    // Disegna una stringa
    void draw(const std::string& text, float x, float y, float scale, const glm::vec3& color);

private:
    Shader shader;
    unsigned int vao = 0;
    unsigned int vbo = 0;
    std::map<char, Character> characters;

    void initBuffers();
    void freeCharacters();
};

#endif
