//  nn.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once
#include "engine.hpp"
#include <random>

template <typename T> T random_uniform(T range_from, T range_to) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

template <typename T> class Neuron {
public:
    Value<T> bias = random_uniform(-1.0, 1.0);
    std::vector<Value<T>> weights;

public:
    // Constructor
    Neuron();

    // Operator
    Value<T> operator()(std::vector<T> x) {
        // Initialize to bias instead of adding it after
        Value<T> weighted_sum = bias;

        // Sum over all multiplies
        for (size_t i = 0; i < x.size(); i++) {
            weighted_sum += (weights[i] * x[i]);
        }
        // Return the activation value of the neuron
        return weighted_sum.tanh();
    }

    void print_graph();
};

// ==================== Implementation =====================

template <typename T> Neuron<T>::Neuron() {
    for (size_t i = 0; i < 2; i++) {
        weights.emplace_back(random_uniform(-1.0, 1.0));
    }
}
