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
    // Constructor
    Neuron(size_t number_of_neurons_input);

    // Call operator: w * x + b dot product
    Value<T> operator()(std::vector<T> x) {
        // Initialize to bias instead of adding it after
        Value<T> weighted_sum = bias;

        // Sum over all multiplies
        for (size_t i = 0; i < m_number_of_neurons_input; i++) {
            weighted_sum += (m_weights[i].data * x[i]);
        }
        // Return the activation value of the neuron as a value object
        return (weighted_sum.tanh());
    }

    void print_graph();

protected:
    Value<T> bias = Value<T>(random_uniform(-1.0, 1.0));
    size_t m_number_of_neurons_input;
    std::vector<Value<T>> m_weights;
};

// ==================== Implementation =====================

template <typename T> Neuron<T>::Neuron(size_t number_of_neurons_input)
    : m_number_of_neurons_input(number_of_neurons_input)
{
    for (size_t i = 0; i < number_of_neurons_input; i++) {
        m_weights.emplace_back(random_uniform(-1.0, 1.0));
    }
}
