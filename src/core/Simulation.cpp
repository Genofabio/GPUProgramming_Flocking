#include "custom/Simulation.h"  
#include "custom/ResourceManager.h"
#include "custom/SpriteRenderer.h"

Simulation::Simulation(unsigned int width, unsigned int height)
    : state(SIMULATION_RUNNING)
    , keys()
    , width(width)
    , height(height)
    , boidRender(nullptr)
    , cohesionDistance(500.0f)
    , separationDistance(50.0f)
    , alignmentDistance(50.0f)
    , borderDistance(250.0f)
    , cohesionScale(0.08f)
    , separationScale(1.5f)
    , alignmentScale(0.15f)
    
{
}

Simulation::~Simulation()
{
    delete boidRender;
}

void Simulation::init()
{
    // caricamento shaders
    ResourceManager::LoadShader("shaders/boid_shader.vert", "shaders/boid_shader.frag", nullptr, "boid");

    glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(this->width),
        static_cast<float>(this->height), 0.0f, -1.0f, 1.0f);

    // impostazione shaders
    ResourceManager::GetShader("boid").Use().SetMatrix4("projection", projection);

    // inizializzazione renderers
    boidRender = new BoidRenderer(ResourceManager::GetShader("boid"));

    // inizializzazione dei boids con posizioni e velocità random
    const int NUM_PREY = 180;
    const int NUM_PREDATORS = 3;

    for (int i = 0; i < NUM_PREY; i++) {
        Boid b;
        b.position = glm::vec2(rand() % width, rand() % height);
        b.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 100,
            (((rand() % 200) - 100) / 100.0f) * 100);
        b.type = PREY;
        boids.push_back(b);
    }

    for (int i = 0; i < NUM_PREDATORS; i++) {
        Boid b;
        b.position = glm::vec2(rand() % width, rand() % height);
        b.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 100,
            (((rand() % 200) - 100) / 100.0f) * 100);
        b.type = PREDATOR;
        boids.push_back(b);
    }
}

void Simulation::update(float dt) {
    std::vector<glm::vec2> velocityChanges(boids.size(), glm::vec2(0.0f));
    float slowDown = 0.3f; 

    // 1. Calcola i cambiamenti di velocità da cohesion, separation, alignment
    for (size_t i = 0; i < boids.size(); i++) {
        Boid& b = boids[i];
        glm::vec2 v(0.0f);

        if (b.type == PREY) {
            // Flocking base
            glm::vec2 v1 = moveTowardCenter(i);
            glm::vec2 v2 = avoidNeighbors(i);
            glm::vec2 v3 = matchVelocity(i);

            // Scappa dai predatori
            glm::vec2 evade = evadePredators(i); // <-- da implementare

            v = v1 + v2 + v3 + evade;
        }
        else if (b.type == PREDATOR) {
            // Inseguire le prede
            glm::vec2 hunt = chasePrey(i);

            // Separazione dai altri predatori
            glm::vec2 separation = avoidOtherPredators(i) * 2.0f;

            v = hunt + separation;
        }

        // Evita i bordi (vale per tutti)
        glm::vec2 borderForce = avoidBorders(b);

        velocityChanges[i] = v + borderForce;
    }

    // 2. Applica i cambiamenti e stabilizza la velocità
    for (size_t i = 0; i < boids.size(); i++) {
        // applica le regole
        boids[i].velocity += velocityChanges[i] * slowDown;

        // clamp della velocità
        float maxSpeed = 150.0f;
        float speed = glm::length(boids[i].velocity);
        if (speed > maxSpeed) {
            boids[i].velocity = (boids[i].velocity / speed) * maxSpeed;
        }

        // 3. aggiorna la posizione
        boids[i].position += boids[i].velocity * dt;
    }
}

void Simulation::processInput(float dt)
{
}

void Simulation::render() {
    for (Boid& b : boids) {
        float angle = glm::degrees(atan2(b.velocity.y, b.velocity.x)) + 270.0f;

        glm::vec3 color;
        if (b.type == PREDATOR) {
            color = glm::vec3(0.9f, 0.2f, 0.2f);   // rosso (predatori)
        }
        else {
            color = glm::vec3(0.4f, 0.5f, 0.9f);   // blu (prede)
        }

        boidRender->DrawBoid(b.position, angle, color, 10.0f);
    }
}

// Cohesion rule
glm::vec2 Simulation::moveTowardCenter(size_t i) {
    glm::vec2 perceived_center(0.0f);
    int count = 0;
    Boid& self = boids[i];

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREY) continue;  // ignora predatori

        float dist = glm::length(boids[j].position - self.position);
        if (dist < cohesionDistance) {
            perceived_center += boids[j].position;
            count++;
        }
    }

    if (count > 0)
        perceived_center = (perceived_center / (float)count - self.position) * cohesionScale;
    else
        perceived_center = glm::vec2(0.0f);

    return perceived_center;
}


// Separation rule
glm::vec2 Simulation::avoidNeighbors(size_t i) {
    glm::vec2 c(0.0f);

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        glm::vec2 diff = boids[i].position - boids[j].position;
        float dist = glm::length(diff);
        if (dist < separationDistance && dist > 0.0f) {
            c += diff / dist; // respinta proporzionale
        }
    }

    return c * separationScale;
}

glm::vec2 Simulation::avoidOtherPredators(size_t i) {
    glm::vec2 c(0.0f);

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREDATOR) continue;
        glm::vec2 diff = boids[i].position - boids[j].position;
        float dist = glm::length(diff);
        if (dist < separationDistance && dist > 0.0f) {
            c += diff / dist; // respinta proporzionale
        }
    }

    return c * separationScale;
}

// Alignment rule
glm::vec2 Simulation::matchVelocity(size_t i) {
    glm::vec2 perceived_velocity(0.0f);
    int count = 0;
    Boid& self = boids[i];

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREY) continue;  // ignora predatori

        float dist = glm::length(boids[j].position - self.position);
        if (dist < alignmentDistance) {
            perceived_velocity += boids[j].velocity;
            count++;
        }
    }

    if (count > 0)
        perceived_velocity = (perceived_velocity / (float)count) * alignmentScale;
    else
        perceived_velocity = glm::vec2(0.0f);

    return perceived_velocity;
}


glm::vec2 Simulation::avoidBorders(const Boid& b) {
    glm::vec2 force(0.0f);

    float distLeft = b.position.x;
    float distRight = width - b.position.x;
    float distTop = b.position.y;
    float distBottom = height - b.position.y;

    if (distLeft < borderDistance)
        force += glm::vec2(1, 0) * (borderDistance - distLeft);
    if (distRight < borderDistance)
        force += glm::vec2(-1, 0) * (borderDistance - distRight);
    if (distTop < borderDistance)
        force += glm::vec2(0, 1) * (borderDistance - distTop);
    if (distBottom < borderDistance)
        force += glm::vec2(0, -1) * (borderDistance - distBottom);

    // Scala finale per evitare scatti troppo forti
    force *= 0.2f;

    return force;
}

glm::vec2 Simulation::evadePredators(size_t i) {
    glm::vec2 c(0.0f);
    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREDATOR) continue;

        glm::vec2 diff = boids[i].position - boids[j].position;
        float dist = glm::length(diff);
        float fearDistance = 150.0f; // distanza di “paura”
        if (dist < fearDistance && dist > 0.0f) {
            c += diff / dist * (fearDistance - dist);
        }
    }
    return c * 0.8f; // forza di fuga
}

glm::vec2 Simulation::chasePrey(size_t i) {
    glm::vec2 target(0.0f);
    float closest = 999999.0f;
    for (size_t j = 0; j < boids.size(); j++) {
        if (boids[j].type != PREY) continue;

        float dist = glm::length(boids[j].position - boids[i].position);
        if (dist < closest) {
            closest = dist;
            target = boids[j].position;
        }
    }
    if (closest < 999999.0f)
        return (target - boids[i].position) * 0.1f; // forza verso la preda
    return glm::vec2(0.0f);
}

//glm::vec2 Simulation::chasePrey(size_t i) {
//    float flockRadius = 100.0f; // distanza per considerare il gruppo
//    glm::vec2 target(0.0f);
//    int maxCount = 0;
//    Boid& predator = boids[i];
//
//    for (size_t j = 0; j < boids.size(); j++) {
//        if (boids[j].type != PREY) continue;
//
//        glm::vec2 center(0.0f);
//        int count = 0;
//
//        for (size_t k = 0; k < boids.size(); k++) {
//            if (boids[k].type != PREY) continue;
//
//            float dist = glm::length(boids[j].position - boids[k].position);
//            if (dist < flockRadius) {
//                center += boids[k].position;
//                count++;
//            }
//        }
//
//        if (count > maxCount) {
//            maxCount = count;
//            target = center / (float)count;
//        }
//    }
//
//    if (maxCount > 0)
//        return (target - predator.position) * 0.2f;
//    return glm::vec2(0.0f);
//}

//glm::vec2 Simulation::chasePrey(size_t i) {
//    Boid& predator = boids[i];
//    glm::vec2 weightedPos(0.0f);
//    glm::vec2 weightedVel(0.0f);
//    float totalWeight = 0.0f;
//    float influenceRadius = 150.0f;
//
//    for (size_t j = 0; j < boids.size(); j++) {
//        if (boids[j].type != PREY) continue;
//
//        glm::vec2 diff = boids[j].position - predator.position;
//        float dist = glm::length(diff);
//        if (dist > 0 && dist < influenceRadius) {
//            float weight = 1.0f / dist; // prede più vicine pesano di più
//            weightedPos += boids[j].position * weight;
//            weightedVel += boids[j].velocity * weight;
//            totalWeight += weight;
//        }
//    }
//
//    if (totalWeight > 0) {
//        glm::vec2 avgPos = weightedPos / totalWeight;
//        glm::vec2 avgVel = weightedVel / totalWeight;
//        return ((avgPos + avgVel) - predator.position) * 0.1f;
//    }
//
//    return glm::vec2(0.0f);
//}