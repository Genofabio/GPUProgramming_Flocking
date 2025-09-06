#include "custom/Simulation.h"  
#include "custom/ResourceManager.h"
#include "custom/SpriteRenderer.h"
#include <iostream>

Simulation::Simulation(unsigned int width, unsigned int height)
    : state(SIMULATION_RUNNING)
    , keys()
    , width(width)
    , height(height)
    , boidRender(nullptr)
    // distances defaults
    , cohesionDistance(200.0f)
    , separationDistance(30.0f)
    , alignmentDistance(50.0f)
    , borderDistance(300.0f)
    , predatorFearDistance(150.0f)
    , predatorChaseDistance(800.0f)
    , predatorSeparationDistance(60.0f)
    // scales / weights defaults
    , cohesionScale(0.05f)
    , separationScale(2.0f)
    , alignmentScale(0.15f)
    , borderScale(0.3f)
    , predatorFearScale(0.8f)
    , predatorChaseScale(0.12f)
    , predatorSeparationScale(2.0f)
    , rng(std::random_device{}()) // Mersenne Twister con seed casuale
    , dist(-1.0f, 1.0f)
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
    const int NUM_PREDATORS = 2;
    const int NUM_LEADERS = 2;

    // Leader
    for (int i = 0; i < NUM_LEADERS; i++) {
        Boid leader;
        leader.position = glm::vec2(rand() % width, rand() % height);
        leader.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 50,
            (((rand() % 200) - 100) / 100.0f) * 50);
        leader.type = LEADER;
        leader.age = 10;
        leader.scale = 1.7f;
        leader.color = glm::vec3(0.9f, 0.9f, 0.2f);
        leader.drift = glm::vec2(0);
        boids.push_back(leader);
    }

    // Non-Leader
    std::uniform_int_distribution<int> ageDist(0, 6);
    std::uniform_real_distribution<float> offsetDist(0.0f, 30.0f);

    for (int i = 0; i < NUM_PREY; i++) {
        Boid b;
        b.position = glm::vec2(rand() % width, rand() % height);
        b.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 100,
            (((rand() % 200) - 100) / 100.0f) * 100);
        b.type = PREY;
        b.birthTime = currentTime + offsetDist(rng); // Birth time random continuo [0, 30]
        b.age = ageDist(rng); // età ranodm [0, 6]
        
        float t = b.age / 10.0f; // Normalizzazione t da 0.0 a 0.6
        
        b.scale = 1.0f + 0.04f * b.age; // Aggiorna scale (1.0 -> 1.24) 

        glm::vec3 blue(0.2f, 0.2f, 0.9f);
        glm::vec3 blue_marine(0.05f, 0.8f, 0.7f);
        b.color = glm::mix(blue, blue_marine, t); // Aggiorna colore (da blu a blu_marine) 

        b.influence = 0.8f + 0.04f * b.age; // Setta influence tra 0.8 (b.age = 0) e 1.04 (b.age = 6) 

        // std::cout << "CurrentTime " << i << currentTime << std::endl;
        // std::cout << "Boid " << i << "birthTime " << b.birthTime << "Age " << b.age << std::endl;

        boids.push_back(b);
    }

    // Predatori
    for (int i = 0; i < NUM_PREDATORS; i++) {
        Boid b;
        b.position = glm::vec2(rand() % width, rand() % height);
        b.velocity = glm::vec2((((rand() % 200) - 100) / 100.0f) * 100,
            (((rand() % 200) - 100) / 100.0f) * 100);
        b.type = PREDATOR;
        b.age = 10;
        b.scale = 1.9f;
        b.color = glm::vec3(0.9f, 0.2f, 0.2f);
        b.drift = glm::vec2(0);
        boids.push_back(b);
    }
}

void Simulation::update(float dt) {
    currentTime += dt;

    std::vector<glm::vec2> velocityChanges(boids.size(), glm::vec2(0.0f));
    float slowDown = 0.3f; 

    // 1. Calcola i cambiamenti di velocità da cohesion, separation, alignment
    for (size_t i = 0; i < boids.size(); i++) {
        Boid& b = boids[i];
        glm::vec2 v(0.0f);

        if (b.type == PREY) {

            upgradeBoid(b, currentTime);

            // Flocking base
            glm::vec2 v1 = moveTowardCenter(i);
            glm::vec2 v2 = avoidNeighbors(i);
            glm::vec2 v3 = matchVelocity(i);
            glm::vec2 evade = evadePredators(i);
            glm::vec2 follow = followLeaders(i);   // <-- nuova regola

            v = v1 + v2 + v3 + evade + follow;
        }
        else if (b.type == PREDATOR) {
            // Inseguire le prede
            glm::vec2 hunt = chasePrey(i);

            // Separazione dai altri predatori
            glm::vec2 separation = avoidOtherPredators(i) * 2.0f;

            v = hunt + separation;
        } else if(b.type == LEADER) {
            glm::vec2 vSep = leaderSeparation(i);
            glm::vec2 evade = evadePredators(i);

            v = vSep + evade;  // possono anche avere velocità “autonome” se vuoi
        }

        // Evita i bordi (vale per tutti)
        glm::vec2 borderForce = avoidBorders(b);

        velocityChanges[i] = v + borderForce;

        float driftChange = 10.0f;   // quanto cambia il drift a ogni frame
        float driftMax = 100.0f;     // massimo contributo random

        b.drift += glm::vec2(dist(rng), dist(rng)) * driftChange * dt;;

        // clamp per non farlo crescere troppo
        if (glm::length(b.drift) > driftMax) {
            b.drift = glm::normalize(b.drift) * driftMax;
        }

        velocityChanges[i] += b.drift;
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

        boidRender->DrawBoid(b.position, angle, b.color, 10.0f*b.scale);
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
            c += diff / dist; // respinta proporzionale (unit vector)
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
        if (dist < predatorSeparationDistance && dist > 0.0f) {
            c += diff / dist; // respinta proporzionale
        }
    }

    return c; // moltiplicato dal chiamante con predatorSeparationScale
}

// Alignment rule
glm::vec2 Simulation::matchVelocity(size_t i) {
    glm::vec2 perceived_velocity(0.0f);
    float totalWeight = 0.0f;
    Boid& self = boids[i];

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREY) continue;  // ignora predatori

        float dist = glm::length(boids[j].position - self.position);
        if (dist < alignmentDistance) {
            float w = boids[j].influence; // peso
            perceived_velocity += boids[j].velocity * w;
            totalWeight += w;
        }
    }

    if (totalWeight > 0.0f)
        perceived_velocity = (perceived_velocity / totalWeight) * alignmentScale;
    else
        perceived_velocity = glm::vec2(0.0f);

    return perceived_velocity;
}

//glm::vec2 Simulation::avoidBorders(const Boid& b) {
//    glm::vec2 force(0.0f);
//
//    float distLeft = b.position.x;
//    float distRight = width - b.position.x;
//    float distTop = b.position.y;
//    float distBottom = height - b.position.y;
//
//    if (distLeft < borderDistance)
//        force += glm::vec2(1, 0) * (borderDistance - distLeft);
//    if (distRight < borderDistance)
//        force += glm::vec2(-1, 0) * (borderDistance - distRight);
//    if (distTop < borderDistance)
//        force += glm::vec2(0, 1) * (borderDistance - distTop);
//    if (distBottom < borderDistance)
//        force += glm::vec2(0, -1) * (borderDistance - distBottom);
//
//    // Scala finale per evitare scatti troppo forti
//    force *= 0.2f;
//
//    return force;
//}

// Borders rule (usa borderDistance e borderScale)
glm::vec2 Simulation::avoidBorders(const Boid& b) {
    glm::vec2 force(0.0f);

    float distLeft = b.position.x;
    float distRight = width - b.position.x;
    float distTop = b.position.y;
    float distBottom = height - b.position.y;

    // lambda che calcola la forza non lineare (stessa forma, ma usa borderDistance)
    auto computeForce = [this](float dist) -> float {
        if (dist < borderDistance)
            return pow(borderDistance - dist, 2) / borderDistance;
        return 0.0f;
        };

    force += glm::vec2(1, 0) * computeForce(distLeft);
    force += glm::vec2(-1, 0) * computeForce(distRight);
    force += glm::vec2(0, 1) * computeForce(distTop);
    force += glm::vec2(0, -1) * computeForce(distBottom);

    // Scala finale: usa borderScale
    force *= borderScale;

    return force;
}

// Evade predators (usa predatorFearDistance e predatorFearScale)
glm::vec2 Simulation::evadePredators(size_t i) {
    glm::vec2 c(0.0f);
    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != PREDATOR) continue;

        glm::vec2 diff = boids[i].position - boids[j].position;
        float dist = glm::length(diff);
        if (dist < predatorFearDistance && dist > 0.0f) {
            // respinta che aumenta quando il predatore è più vicino
            c += (diff / dist) * (predatorFearDistance - dist);
        }
    }
    return c * predatorFearScale;
}

// Chase prey (usa predatorChaseDistance e predatorChaseScale)
glm::vec2 Simulation::chasePrey(size_t i) {
    glm::vec2 target(0.0f);
    float closest = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < boids.size(); j++) {
        if (boids[j].type != PREY) continue;

        float dist = glm::length(boids[j].position - boids[i].position);
        // considera solo prede entro predatorChaseDistance
        if (dist < closest && dist < predatorChaseDistance) {
            closest = dist;
            target = boids[j].position;
        }
    }
    if (closest < std::numeric_limits<float>::infinity()) {
        // forza diretta verso la preda
        glm::vec2 dir = target - boids[i].position;
        // normalizza e scala per evitare scatti troppo forti se molto vicino
        float len = glm::length(dir);
        if (len > 0.0f) {
            return (dir / len) * ((predatorChaseDistance - closest) / predatorChaseDistance) * predatorChaseScale * 100.0f;
        }
    }
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

glm::vec2 Simulation::followLeaders(size_t i) {
    Boid& self = boids[i];

    glm::vec2 towardLeader(0.0f);
    float closestDist = std::numeric_limits<float>::infinity();
    glm::vec2 closestLeaderPos(0.0f);

    float leaderInfluenceDistance = 200.0f; // regolabile

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != LEADER) continue;  // <-- fix confronto

        float dist = glm::length(boids[j].position - self.position);
        if (dist < leaderInfluenceDistance && dist < closestDist) {
            closestDist = dist;
            closestLeaderPos = boids[j].position;
        }
    }

    if (closestDist < std::numeric_limits<float>::infinity()) {
        float norm = (leaderInfluenceDistance - closestDist) / leaderInfluenceDistance;
        // curva dolce: vicino = peso quasi nullo, lontano = attrazione maggiore
        float weight = norm * norm; // quadratica, abbassa la forza a corto raggio
        towardLeader = (closestLeaderPos - self.position) * weight * 0.4f;
    }

    return towardLeader; // 0 se nessun leader vicino
}

glm::vec2 Simulation::leaderSeparation(size_t i) {
    glm::vec2 force(0.0f);
    Boid& self = boids[i];

    for (size_t j = 0; j < boids.size(); j++) {
        if (i == j) continue;
        if (boids[j].type != LEADER) continue;

        glm::vec2 diff = self.position - boids[j].position;
        float dist = glm::length(diff);
        float desiredLeaderDistance = 200.0f; // regolabile
        if (dist < desiredLeaderDistance && dist > 0.0f) {
            force += diff / dist * (desiredLeaderDistance - dist);
        }
    }

    return force * 0.8f; // scala la forza
}

// Aggiorna i parametri di crescita della preda
void Simulation::upgradeBoid(Boid& b, float currentTime) {
    if (b.type != PREY || b.age >= 10) return;  // solo PREY e max età 10

    if (currentTime - b.birthTime >= 15.0f) { // iniziano a crescere dopo 15s
        //std::cout << "upgrade al tempo " << currentTime << ". Age: " << b.age << " -> " << b.age +1 << std::endl;
        b.age++;  // incremento età

        b.scale += 0.04f; // Aggiorna scale (1.0 -> 1.4)

        // Aggiorna colore (da blu a blu_marine)
        float t = b.age / 10.0f; // Normalizzazione t da 0.0 a 1.0

        glm::vec3 blue(0.2f, 0.2f, 0.9f);
        glm::vec3 blue_marine(0.05f, 0.8f, 0.7f);
        b.color = glm::mix(blue, blue_marine, t);

        // Aggiorna influence (0.8 -> 1.2)
        b.influence += 0.04f;

        // Reset birthTime per prossimo step
        b.birthTime = currentTime;
    }
}
