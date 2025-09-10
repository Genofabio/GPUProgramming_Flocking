#include <limits>
#include <cmath>
#include <external/glm/gtc/matrix_transform.hpp>
#include <external/glm/gtc/type_ptr.hpp>
#include <core/BoidRules.h>

namespace BoidRules {

    glm::vec2 computeCohesion(const Boid& self, const std::vector<Boid>& boids, const std::vector<size_t>& nearbyIndices, float cohesionDistance, float cohesionScale) {
        glm::vec2 perceived_center(0.0f);
        int count = 0;

        for (size_t idx : nearbyIndices) {
            const Boid& b = boids[idx];
            if (b.type != PREY) continue;
            float dist = glm::length(b.position - self.position);
            if (dist < cohesionDistance) {
                perceived_center += b.position;
                count++;
            }
        }

        if (count > 0)
            perceived_center = (perceived_center / float(count) - self.position) * cohesionScale;

        return perceived_center;
    }

    glm::vec2 computeSeparation(const Boid& self, const std::vector<Boid>& boids, const std::vector<size_t>& nearbyIndices, float separationDistance, float separationScale) {
        glm::vec2 c(0.0f);

        for (size_t idx : nearbyIndices) {
            const Boid& other = boids[idx];
            if (&self == &other) continue;
            glm::vec2 diff = self.position - other.position;
            float dist = glm::length(diff);
            if (dist < separationDistance && dist > 0.0f) {
                c += diff / dist;
            }
        }

        return c * separationScale;
    }

    glm::vec2 computePredatorSeparation(const Boid& self, const std::vector<Boid>& boids, const std::vector<size_t>& nearbyIndices, float predatorSeparationDistance) {
        glm::vec2 c(0.0f);

        for (size_t idx : nearbyIndices) {
            const Boid& other = boids[idx];
            if (&self == &other) continue;
            if (other.type != PREDATOR) continue;

            glm::vec2 diff = self.position - other.position;
            float dist = glm::length(diff);
            if (dist < predatorSeparationDistance && dist > 0.0f) {
                c += diff / dist;
            }
        }

        return c;
    }

    glm::vec2 computeAlignment(const Boid& self, const std::vector<Boid>& boids, const std::vector<size_t>& nearbyIndices, float alignmentDistance, float alignmentScale) {
        glm::vec2 perceived_velocity(0.0f);
        float totalWeight = 0.0f;

        for (size_t idx : nearbyIndices) {
            const Boid& other = boids[idx];
            if (&self == &other) continue;
            if (other.type != PREY) continue;

            float dist = glm::length(other.position - self.position);
            if (dist < alignmentDistance) {
                float w = other.influence;
                perceived_velocity += other.velocity * w;
                totalWeight += w;
            }
        }

        if (totalWeight > 0.0f)
            perceived_velocity = (perceived_velocity / totalWeight) * alignmentScale;
        else
            perceived_velocity = glm::vec2(0.0f);

        return perceived_velocity;
    }

    glm::vec2 computeEvadePredators(const Boid& self, const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyPredators, const std::vector<size_t>& nearbyAllies,
        float predatorFearDistance, float predatorFearScale, float allyRadius)
    {
        glm::vec2 c(0.0f);

        for (size_t idx : nearbyPredators) {
            const Boid& other = boids[idx];
            glm::vec2 diff = self.position - other.position;
            float dist = glm::length(diff);
            if (dist < predatorFearDistance && dist > 0.0f) {
                c += (diff / dist) * (predatorFearDistance - dist);
            }
        }

        int nearbyCount = 0;
        for (size_t idx : nearbyAllies) {
            const Boid& other = boids[idx];
            float dist = glm::length(other.position - self.position);
            if (dist < allyRadius) nearbyCount++;
        }

        const float k = 10.03f;
        const float n0 = 2.08f;
        float groupFactor = 1.0f / (1.0f + std::exp(-k * (nearbyCount - n0)));

        return c * predatorFearScale * groupFactor;
    }

    glm::vec2 computeChasePrey(size_t predatorIndex, const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyPrey,
        float predatorChaseDistance, float predatorChaseScale, float predatorBoostRadius)
    {
        glm::vec2 outForce(0.0f);
        glm::vec2 target(0.0f);
        float closest = std::numeric_limits<float>::infinity();
        int nearbyCount = 0;

        const Boid& predator = boids[predatorIndex];

        for (size_t idx : nearbyPrey) {
            const Boid& prey = boids[idx];
            glm::vec2 diff = prey.position - predator.position;
            float dist = glm::length(diff);

            if (dist < closest && dist < predatorChaseDistance) {
                closest = dist;
                target = prey.position;
            }

            if (dist < predatorBoostRadius) nearbyCount++;
        }

        if (closest == std::numeric_limits<float>::infinity()) return glm::vec2(0.0f);

        glm::vec2 dir = target - predator.position;
        float len = glm::length(dir);
        if (len > 0.0f) {
            float baseForce = ((predatorChaseDistance - closest) / predatorChaseDistance) * predatorChaseScale * 100.0f;
            float boost = (nearbyCount <= 2) ? 1.5f : 1.0f;
            outForce = (dir / len) * baseForce * boost;
        }

        return outForce;
    }

    bool computeEatPrey(size_t predatorIndex, size_t preyIndex, const std::vector<Boid>& boids, float predatorEatDistance) {
        if (predatorIndex >= boids.size() || preyIndex >= boids.size()) return false;
        if (boids[predatorIndex].type != PREDATOR || boids[preyIndex].type != PREY) return false;

        float dist = glm::length(boids[predatorIndex].position - boids[preyIndex].position);
        return (dist < predatorEatDistance);
    }

    glm::vec2 computeFollowLeaders(const Boid& self, const std::vector<Boid>& boids, const std::vector<size_t>& nearbyLeaders, float leaderInfluenceDistance) {
        glm::vec2 towardLeader(0.0f);
        float closestDist = std::numeric_limits<float>::infinity();
        glm::vec2 closestLeaderPos(0.0f);

        for (size_t idx : nearbyLeaders) {
            const Boid& leader = boids[idx];
            float dist = glm::length(leader.position - self.position);
            if (dist < leaderInfluenceDistance && dist < closestDist) {
                closestDist = dist;
                closestLeaderPos = leader.position;
            }
        }

        if (closestDist < std::numeric_limits<float>::infinity()) {
            float norm = (leaderInfluenceDistance - closestDist) / leaderInfluenceDistance;
            float weight = norm * norm;
            towardLeader = (closestLeaderPos - self.position) * weight * 0.4f;
        }

        return towardLeader;
    }

    glm::vec2 computeLeaderSeparation(const Boid& self, const std::vector<Boid>& boids, const std::vector<size_t>& nearbyLeaders, float desiredLeaderDistance) {
        glm::vec2 force(0.0f);

        for (size_t idx : nearbyLeaders) {
            const Boid& leader = boids[idx];
            glm::vec2 diff = self.position - leader.position;
            float dist = glm::length(diff);

            if (dist < desiredLeaderDistance && dist > 0.0f) {
                force += diff / dist * (desiredLeaderDistance - dist);
            }
        }

        return force * 0.8f;
    }

    glm::vec2 computeBorderRepulsion(const glm::vec2& position, float width, float height, float borderAlertDistance) {
        glm::vec2 repulsion(0.0f);
        float distLeft = position.x;
        float distRight = width - position.x;
        float distTop = position.y;
        float distBottom = height - position.y;

        if (distLeft < borderAlertDistance)   repulsion += glm::vec2(1, 0) * (borderAlertDistance - distLeft);
        if (distRight < borderAlertDistance)  repulsion += glm::vec2(-1, 0) * (borderAlertDistance - distRight);
        if (distTop < borderAlertDistance)    repulsion += glm::vec2(0, 1) * (borderAlertDistance - distTop);
        if (distBottom < borderAlertDistance) repulsion += glm::vec2(0, -1) * (borderAlertDistance - distBottom);

        repulsion *= 0.2f;
        return repulsion;
    }

    glm::vec2 computeWallRepulsion(const glm::vec2& position, const glm::vec2& velocity, const std::vector<Wall>& walls) {
        glm::vec2 repulsion(0.0f);
        const float lookAhead = 30.0f;
        glm::vec2 dir = glm::normalize(velocity);

        for (const Wall& w : walls) {
            glm::vec2 closestPoint;
            float dist = w.distanceToPoint(position, closestPoint);

            if (dist < w.repulsionDistance && dist > 0.001f) {
                float safeLookAhead = glm::clamp(dist - 0.2f, 0.001f, lookAhead);
                glm::vec2 probePos = position + dir * safeLookAhead;
                glm::vec2 away = glm::normalize(probePos - closestPoint);
                float factor = (w.repulsionDistance - dist) / dist;
                repulsion += away * factor * factor * w.repulsionStrength;
            }
        }

        return repulsion;
    }

    // --- Mating & Boid Upgrade ---
    void computeMating(
        size_t i,
        const std::vector<Boid>& boids,
        std::vector<int>& boidCouples,
        std::vector<std::pair<size_t, size_t>>& spawnPairs,
        float mateDistance,
        int mateThreshold,
        int matingAge
    ) {
        size_t N = boids.size();
        if (i >= N || boids[i].type != PREY) return;

        // Assicurati che boidCouples abbia dimensione N*N
        if (boidCouples.size() < N * N) {
            boidCouples.resize(N * N, 0);
        }

        for (size_t j = i + 1; j < N; j++) {
            if (boids[j].type != PREY) continue;

            float dist = glm::length(boids[i].position - boids[j].position);
            size_t idx = i * N + j;

            if (dist < mateDistance && boids[i].age >= matingAge && boids[j].age >= matingAge) {
                boidCouples[idx]++;

                if (boidCouples[idx] >= mateThreshold) {
                    spawnPairs.emplace_back(i, j); // coppia pronta a spawnare
                    boidCouples[idx] = 0;          // reset contatore
                }
            }
            else {
                boidCouples[idx] = 0; // reset se si separano
            }
        }
    }

    Boid computeSpawnedBoid(
        const Boid& parentA,
        const Boid& parentB,
        float currentTime
    ) {
        Boid b;

        b.position = (parentA.position + parentB.position) / 2.0f;
        b.velocity = (parentA.velocity + parentB.velocity) / 2.0f;
        b.type = PREY;
        b.birthTime = currentTime;
        b.age = 0;
        b.scale = 1.0f;
        b.color = glm::vec3(0.2f, 0.2f, 0.9f); // blu
        b.influence = 0.8f;

        return b;
    }

    void computeBoidUpgrade(
        Boid& b,
        float currentTime
    ) {
        if (b.type != PREY || b.age >= 10) return;

        if (currentTime - b.birthTime >= 15.0f) {
            b.age++;             // incremento età
            b.scale += 0.04f;    // aggiorna scala

            // aggiornamento colore da blu a blu_marine
            float t = b.age / 10.0f;
            glm::vec3 blue(0.2f, 0.2f, 0.9f);
            glm::vec3 blue_marine(0.05f, 0.8f, 0.7f);
            b.color = glm::mix(blue, blue_marine, t);

            // aggiorna influence
            b.influence += 0.04f;

            // reset birthTime
            b.birthTime = currentTime;
        }
    }


} // namespace BoidRules