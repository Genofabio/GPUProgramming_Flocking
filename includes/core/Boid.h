enum BoidType { PREY, PREDATOR, LEADER };

struct Boid {
    BoidType type;          // Tipo del boid (PREY, PREDATOR, LEADER)

    glm::vec2 position;     // Posizione corrente nel mondo 2D
    glm::vec2 velocity;     // Velocità corrente del boid
    glm::vec2 drift;        // Deriva casuale o forze extra (usata per leader/predatori)

    float scale;            // Scala per il rendering (grandezza del boid)
    float influence;        // Influenza sugli altri boid (usata nelle regole di coesione/allineamento)

    glm::vec3 color;        // Colore del boid (RGB)

    int age;                // Età del boid (discreta, usata per regole come matingAge)
    float birthTime;        // Tempo di nascita (float per interpolazioni temporali, accoppiamenti, ecc.)
};