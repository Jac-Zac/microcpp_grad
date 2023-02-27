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

// THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION BUT SAVES ALL THE INTERMEDIATE
// STATES TO DRAW THE GRAPH

// ---------------------------------------------------------

template <typename T> class Neuron {
public:
    Neuron(size_t num_neurons_input);

    // Call operator: w * x + b dot product
    Value<T> operator()(std::vector<Value<T>> x);

public:
    size_t m_num_neurons_input;
    // Save the return value so that I can draw the graph
    std::vector<Value<T>> m_weights;
    // weighted_sum already initialized with the bias
    Value<T> m_weighted_sum = Value<T>(0);
    Value<T> bias = Value<T>(random_uniform(-1.0, 1.0));
};

// ---------------------------------------------------------

template <typename T> class Layer {
public:
    Layer(size_t num_neurons_input, size_t num_neurons_out);

    // Call operator: w * x + b dot product
    std::vector<Value<T>> operator()(std::vector<Value<T>> x);

public:
    std::vector<Neuron<T>> m_neurons;
    // Create an array of neurons to return
    std::vector<Value<T>> m_neurons_output;
};

// ----------------------------------------------------------

template <typename T, size_t N> class MLP {
public:
    MLP(size_t num_neurons_input, std::array<size_t, N> num_neurons_out);

    // Call operator: w * x + b dot product
    std::vector<Value<T>> operator()(std::vector<Value<T>> x);

    // << operator overload to get the structure of the network
    std::ostream &operator<<(std::ostream &os);

public:
    const size_t m_num_neurons_in;
    const std::array<size_t, N> m_num_neurons_out;

public:
    std::vector<Layer<T>> m_layers;
};

//  ================ Implementation  Neuron =================

template <typename T>
Neuron<T>::Neuron(size_t number_of_neurons_input)
    : m_num_neurons_input(number_of_neurons_input) {
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        m_weights.emplace_back(Value<T>(random_uniform(-1.0, 1.0)));
    }
}

template <typename T> Value<T> Neuron<T>::operator()(std::vector<Value<T>> x) {

    // Sum over all multiplies
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        m_weighted_sum +=  (m_weights[i] * x[i]);
    }

    m_weighted_sum += bias;

    // Return the activation value of the neuron as a value object
    return (m_weighted_sum.tanh());
}

//  ================ Implementation  Layer =================

template <typename T>
Layer<T>::Layer(size_t num_neurons_input, size_t num_neurons_output) {
    for (size_t i = 0; i < num_neurons_output; i++) {
        m_neurons.emplace_back(Neuron<T>(num_neurons_input));
    }
}

template <typename T>
std::vector<Value<T>> Layer<T>::operator()(std::vector<Value<T>> x) {

    // Iterate over the m_neurons
    for (auto &neuron : m_neurons) {
        m_neurons_output.emplace_back(neuron(x));
    }
    return m_neurons_output;
}

//  ================ Implementation MLP =================

template <typename T, size_t N>
MLP<T, N>::MLP(size_t num_neurons_input,
               std::array<size_t, N> num_neurons_output)
    : m_num_neurons_in(num_neurons_input),
      m_num_neurons_out(num_neurons_output) {

    // Create the first layer with the input neuron size
    m_layers.emplace_back(Layer<T>(num_neurons_input, num_neurons_output[0]));

    // Create the following layers
    for (size_t i = 1; i < N; i++) {
        // Create layers with
        m_layers.emplace_back(
            Layer<T>(num_neurons_output[i - 1], num_neurons_output[i]));
    }
}

template <typename T, size_t N>
std::vector<Value<T>> MLP<T, N>::operator()(std::vector<Value<T>> x) {

    std::vector<Value<T>> layer_output = x;

    for (Layer<T> layer : m_layers) {
        layer_output = layer(layer_output);
    }

    return layer_output;
}

// Overloading for the output to standard out ---------------------------

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &os, const MLP<T, N> &mlp) {
    os << "Network of " << N + 1 << " Layers: [ " << mlp.m_num_neurons_in;
    for (size_t i = 0; i < N; i++) {
        os << " , " << mlp.m_num_neurons_out[i];
    }
    os << " ]\n";
    return os;
}
