#include "sampler.h"
#include <cmath>
#include <algorithm>

Sampler::Sampler(float temperature, float top_p)
    : temperature_(temperature), top_p_(top_p), rng_(std::random_device{}()) {}

int Sampler::sample(const std::vector<float>& logits) {
    int vocab_size = logits.size();

    float inv_temp = (temperature_ > 0.0f) ? (1.0f / temperature_) : 1.0f;
    bool apply_temp = (temperature_ != 1.0f && temperature_ > 0.0f);

    float max_logit = -1e9f;
    if (apply_temp) {
        for (int i = 0; i < vocab_size; i++) {
            float val = logits[i] * inv_temp;
            if (val > max_logit) max_logit = val;
        }
    } else {
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
    }

    std::vector<std::pair<float, int>> probs(vocab_size);
    float sum_exp = 0.0f;
    if (apply_temp) {
        for (int i = 0; i < vocab_size; i++) {
            float p = std::exp((logits[i] * inv_temp) - max_logit);
            probs[i] = {p, i};
            sum_exp += p;
        }
    } else {
        for (int i = 0; i < vocab_size; i++) {
            float p = std::exp(logits[i] - max_logit);
            probs[i] = {p, i};
            sum_exp += p;
        }
    }

    for (int i = 0; i < vocab_size; i++) {
        probs[i].first /= sum_exp;
    }

    if (top_p_ > 0.0f && top_p_ < 1.0f) {
        std::sort(probs.begin(), probs.end(), [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        });

        float cumsum = 0.0f;
        int last_idx = 0;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i].first;
            last_idx = i;
            if (cumsum >= top_p_) break;
        }
        probs.resize(last_idx + 1);
        for (int i = 0; i <= last_idx; i++) {
            probs[i].first /= cumsum;
        }
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);

    float cur_sum = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        cur_sum += probs[i].first;
        if (r <= cur_sum) return probs[i].second;
    }
    return probs.back().second;
}
