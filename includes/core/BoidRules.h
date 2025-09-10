#pragma once
#include <vector>
#include <external/glm/glm.hpp>

#include <core/Boid.h>
#include <core/Wall.h>

namespace BoidRules {

    // === Funzioni per le prede ===
    glm::vec2 computeCohesion(
        const Boid& self,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyIndices,
        float cohesionDistance,
        float cohesionScale
    );

    glm::vec2 computeSeparation(
        const Boid& self,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyIndices,
        float separationDistance,
        float separationScale
    );

    glm::vec2 computeAlignment(
        const Boid& self,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyIndices,
        float alignmentDistance,
        float alignmentScale
    );

    glm::vec2 computeEvadePredators(
        const Boid& self,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyPredators,
        const std::vector<size_t>& nearbyAllies,
        float predatorFearDistance,
        float predatorFearScale,
        float allyRadius
    );

    void computeMating(
        size_t i,
        const std::vector<Boid>& boids,
        std::vector<int>& boidCouples,
        std::vector<std::pair<size_t, size_t>>& spawnPairs,
        float mateDistance,
        int mateThreshold,
        int matingAge
    );

    Boid computeSpawnedBoid(
        const Boid& parentA,
        const Boid& parentB,
        float currentTime
    );

    void computeBoidUpgrade(
        Boid& b,
        float currentTime
    );

    // === Funzioni per i leader ===
    glm::vec2 computeFollowLeaders(
        const Boid& self,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyLeaders,
        float leaderInfluenceDistance
    );

    glm::vec2 computeLeaderSeparation(
        const Boid& self,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyLeaders,
        float desiredLeaderDistance
    );

    // === Funzioni per i predatori ===
    glm::vec2 computeChasePrey(
        size_t predatorIndex,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyPrey,
        float predatorChaseDistance,
        float predatorChaseScale,
        float predatorBoostRadius
    );

    bool computeEatPrey(
        size_t predatorIndex,
        size_t preyIndex,
        const std::vector<Boid>& boids,
        float predatorEatDistance
    );

    glm::vec2 computePredatorSeparation(
        const Boid& self,
        const std::vector<Boid>& boids,
        const std::vector<size_t>& nearbyIndices,
        float predatorSeparationDistance
    );

    // === Funzioni generali ===
    glm::vec2 computeBorderRepulsion(
        const glm::vec2& position,
        float width,
        float height,
        float borderAlertDistance
    );

    glm::vec2 computeWallRepulsion(
        const glm::vec2& position,
        const glm::vec2& velocity,
        const std::vector<Wall>& walls
    );

} // namespace BoidRules
