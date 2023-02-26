//  nn.hppT
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

// ---------------------------------------------------------

template <typename T> class Neuron {
public:
    // Constructor
    Neuron(size_t num_neurons_input);

    // Call operator: w * x + b dot product
    Value<T> operator()(std::vector<T> x);

    void print_graph();

protected:
    Value<T> bias = Value<T>(random_uniform(-1.0, 1.0));
    size_t m_num_neurons_input;
    std::vector<Value<T>> m_weights;
};

// ---------------------------------------------------------

template <typename T> class Layer {
public:
    // Constructor
    Layer(size_t num_neurons_input, size_t num_neurons_out);

    // Call operator: w * x + b dot product
    std::vector<Value<T>> operator()(std::vector<T> x);

    void print_graph();

protected:
    size_t m_num_neurons_input;
    size_t m_num_neurons_output;
    std::vector<Neuron<T>> m_neurons;
};

// ----------------------------------------------------------

//  ================ Implementation  Neuron =================

template <typename T> Neuron<T>::Neuron(size_t number_of_neurons_input)
    : m_num_neurons_input(number_of_neurons_input)
{
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        m_weights.emplace_back(random_uniform(-1.0, 1.0));
    }
}

template <typename T>
Value<T> Neuron<T>::operator()(std::vector<T> x) {
    // Initialize to bias instead of adding it after
    Value<T> weighted_sum = bias;

    // Sum over all multiplies
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        weighted_sum += (m_weights[i].data * x[i]);
    }
    // Return the activation value of the neuron as a value object
    return (weighted_sum.tanh());
}

//  ================ Implementation  Layer =================

template <typename T> Layer<T>::Layer(size_t num_neurons_input, size_t num_neurons_output)
    : m_num_neurons_input(num_neurons_input), m_num_neurons_output(num_neurons_output)
{
    for(size_t i = 0; i < m_num_neurons_output ; i++){
        m_neurons.emplace_back(Neuron<T>(m_num_neurons_input));
    }
}

template <typename T>
std::vector<Value<T>> Layer<T>::operator()(std::vector<T> x) {
    // Create an array of neurons to return
    std::vector<Value<T>> neurons_output;

    // Iterate over the m_neurons
    for(auto &neuron : m_neurons){
        neurons_output.emplace_back(neuron(x));
    }
    return neurons_output;
}
