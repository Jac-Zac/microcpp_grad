//  nn.hppT
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once

#include "engine.hpp"
#include <random>
#include <variant>

template <typename T> T random_uniform(T range_from, T range_to) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

// THIS IS NOT THE MOST EFFICIENT IMPLEMENTATION BUT SAVES ALL THE INTERMEDIATE
// STATES TO DRAW THE GRAPH

// I have to propagate the gradient trough the bias

// ---------------------------------------------------------

// Module Parent class as an interface
template <typename T>
class Module{
public:
    void zero_grad(){
        for (auto& p: parameters()){
            p->grad = 0.0;
        }
    }

    // Make it virtual so that it can be override
    virtual std::vector<Value<T>*> parameters(){
        return {};
    }
};

template <typename T> class Neuron : public Module<T> {
public:
    Neuron(size_t num_neurons_input);

    // Call operator: w * x + b dot product
    Value<T> operator()(std::vector<Value<T>> &x);
    // Overriding
    virtual std::vector<Value<T>*> parameters() override;

public:
    size_t m_num_neurons_input;
    std::vector<Value<T>> m_weights;
    // I'm not propagating the gradient to the bias
    Value<T> m_bias;
};

// ---------------------------------------------------------

template <typename T> class Layer : public Module<T> {
public:
    Layer(size_t num_neurons_input, size_t num_neurons_out);

    // Call operator: w * x + b dot product
    std::vector<Value<T>> operator()(std::vector<Value<T>> &x);
    // Overriding
    virtual std::vector<Value<T>*> parameters() override;

protected:
    // Create the neurons for the layer
    std::vector<Neuron<T>> m_neurons;
};

// ----------------------------------------------------------

template <typename T, size_t N> class MLP : public Module<T> {
public:
    MLP(size_t num_neurons_input, std::array<size_t, N> num_neurons_out);

    // Call operator: w * x + b dot product
    std::vector<std::vector<Value<T>>>* operator()(std::vector<Value<T>> &x);

    // << operator overload to get the structure of the network
    std::ostream &operator<<(std::ostream &os);
    // Overriding
    virtual std::vector<Value<T>*> parameters() override;

public:
    std::vector<Layer<T>> m_layers;
    size_t m_net_size;
    // Layer given in output
protected:
    const size_t m_num_neurons_in;
    // N layers of the N + 1 total have outputs
    const std::array<size_t, N> m_num_neurons_out;
};

//  ================ Implementation  Neuron =================

template <typename T>
Neuron<T>::Neuron(size_t number_of_neurons_input)
    : m_num_neurons_input(number_of_neurons_input),
      m_bias(random_uniform(-1.0, 1.0)) {
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        m_weights.emplace_back(Value<T>(random_uniform(-1.0, 1.0), "weight"));
    }
}

template <typename T> Value<T> Neuron<T>::operator()(std::vector<Value<T>> &x) {

    // Save the result on the heap to use it later when I need it
    /* Value<T>* m_weighted_sum = new Value<T>(0.0, "Neuron_output"); */
    Value<T>* m_weighted_sum = new Value<T>(0.0, "Neuron_output");

    // Sum over all multiplies
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        *m_weighted_sum += m_weights[i] * x[i];
    }

    // Add the bias
    *m_weighted_sum += m_bias;

    // Return the activation value of the neuron as a value object
    return m_weighted_sum->tanh();
}

template <typename T>
std::vector<Value<T>*> Neuron<T>::parameters() {
    // Create a vector for the pointers to the parameters to modici them directly
    std::vector<Value<T>*> params;
    for (auto& w : m_weights) {
        // Add the weights
        params.push_back(&w);
    }
    // Add the biases
    params.push_back(&m_bias);
    return params;
}

//  ================ Implementation  Layer =================

template <typename T>
Layer<T>::Layer(size_t num_neurons_input, size_t num_neurons_output) {
    // Add all the neurons to the layer by crating them
    for (size_t i = 0; i < num_neurons_output; i++) {
        m_neurons.emplace_back(Neuron<T>(num_neurons_input));
    }
}

template <typename T>
std::vector<Value<T>> Layer<T>::operator()(std::vector<Value<T>> &x) {
    // Create an array of neurons to return
    std::vector<Value<T>> m_neurons_output;

    // Iterate over the m_neurons
    for (auto &neuron : m_neurons) {
        m_neurons_output.emplace_back(neuron(x));
    }
    return m_neurons_output;
}

template <typename T>
std::vector<Value<T>*> Layer<T>::parameters() {
    std::vector<Value<T>*> params;
    // Iterate over all the neurons
    for (auto& neuron : m_neurons) {
        auto neuron_params = neuron.parameters();
        // insert an object at the end thus we are not using emplace_back
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());

    }
    return params;
}

//  ================ Implementation MLP =================

template <typename T, size_t N>
MLP<T, N>::MLP(size_t num_neurons_input,
               std::array<size_t, N> num_neurons_output)
    : m_num_neurons_in(num_neurons_input),
      m_num_neurons_out(num_neurons_output)
{

    // Create the first layer with the input neuron size
    m_layers.emplace_back(Layer<T>(num_neurons_input, num_neurons_output[0]));

    // Create the following layers
    for (size_t i = 1; i < N; i++) {
        // Create layers N layers with the number of neuron from the previous layers and output as the current
        m_layers.emplace_back(
            Layer<T>(num_neurons_output[i - 1], num_neurons_output[i]));
    }
}

template <typename T, size_t N>
std::vector<std::vector<Value<T>>>* MLP<T, N>::operator()(std::vector<Value<T>> &x) {

    // Create a new value of value
    std::vector<std::vector<Value<T>>>* single_layer_output = new std::vector<std::vector<Value<T>>>;

    // Set the current layer to the given values
    single_layer_output->emplace_back(x);

    // Iterate over the layer from the second one to the N + 1 layer
    for (size_t i = 1; i <= N; i++) {
        single_layer_output->emplace_back(
            m_layers[i - 1]((*single_layer_output)[i - 1]));
    }

    // Return a Value<T> if it is just one and else return the array of values
    return single_layer_output;
}

template <typename T, size_t N>
std::vector<Value<T>*> MLP<T,N>::parameters() {
    std::vector<Value<T>*> params;
    // Iterate over all the layers
    for (auto& layer : m_layers) {
        auto layer_params = layer.parameters();
        // insert an object at the end thus we are not using emplace_back
        params.insert(params.end(), layer_params.begin(), layer_params.end());

    }
    return params;
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
