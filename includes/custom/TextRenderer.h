#ifndef TEXT_RENDERER_H
#define TEXT_RENDERER_H

#include <map>
#include <string>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <glad/glad.h>
#include <glm/glm.hpp>

#include "shader.h"

struct Character {
    unsigned int TextureID;  // ID della texture glyph
    glm::ivec2   Size;       // Dimensione del glyph
    glm::ivec2   Bearing;    // Offset dal baseline all'inizio del glyph
    unsigned int Advance;    // Offset orizzontale per il prossimo carattere
};

class TextRenderer {
public:
    std::map<char, Character> Characters;

    // Costruttore
    TextRenderer(const Shader& shader);

    // Inizializza FreeType e carica il font
    void Use(std::string font, unsigned int fontSize);

    // Renderizza una stringa
    void RenderText(std::string text, float x, float y, float scale, glm::vec3 color);

private:
    Shader       shader;
    unsigned int VAO, VBO;

    // Setup OpenGL per il rendering del testo
    void initRenderData();
};

#endif
