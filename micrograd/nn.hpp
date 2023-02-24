//  nn.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once
#include "engine.hpp"
#include <random>

template <size_t N> class Neuron {
public:
    double bias;
    std::array<double, N> weights;

public:
    // Constructor
    Neuron();
    // Operator

    double operator()(std::array<double, N> x) { return 0.0; }

    void print_graph();
};

template <typename T> T random_uniform(T range_from, T range_to) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

// ==================== Implementation =====================

template <size_t N> Neuron<N>::Neuron() {
    for (size_t j = 0; j < N; j++) {
        weights[j] = random_uniform(-1.0, 1.0);
    }
    bias = random_uniform(-1.0, 1.0);
}
