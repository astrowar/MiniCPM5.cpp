#pragma once

#include <vector>
#include <random>

// ============================================================================
// STOCHASTIC SAMPLER
//
// Encapsulates the stochastic token sampling logic (Top-p & Temperature).
// Keeping this isolated allows the generation engine to cleanly pull new tokens
// without cluttering the main conversation flow.
// ============================================================================

class Sampler {
public:
    Sampler(float temperature = 1.0f, float top_p = 0.95f);

    // Samples a single token from the logits distribution
    int sample(const std::vector<float>& logits);

private:
    float temperature_;
    float top_p_;
    std::mt19937 rng_;
};
