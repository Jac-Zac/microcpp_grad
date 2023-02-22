//
//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#pragma once

#include <array>
#include <cmath>
#include <stack>
#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <unordered_set>

// Instead of using string use an enum the implementation of different op

template <typename T> class Value {
public:
    T data;             // data of the value
    T grad;             // gradient which by default is zero
    std::string label;  // label of the value
public:
    // Constructor
    Value(T data, std::string label = "", std::string op = "",
          std::array<Value<T> *, 2> children = {});

    // Operator Overloading
    Value operator+(Value &other);
    Value operator*(Value &other);
    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value<T> &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad
           << ", label=" << v.label << ")";
        return os;
    };

    // Other ops
    Value tanh();
    // Value relu();

    void backprop();
    void print_graph();

protected:
    std::string m_op;
    std::array<Value<T> *, 2> m_prev; // previous values
protected:
    // Helper function to make a topological sort
    void topo_sort_helper(Value<T>* v, std::vector<Value<T>*>& sorted_values);
    void m_backward(); // 1 step of backdrop
};

// ==================== Implementation =====================

template <typename T>
Value<T>::Value(T data, std::string label, std::string op,
                std::array<Value<T> *, 2> children)
    : data(data), label(label), m_op(op), m_prev(children), grad(0) {}

template <typename T>
void Value<T>::m_backward(){
    if(this->m_op == "+"){
        // Should just move the gradient along to both of them
        this->m_prev[0]->grad = 1.0 * this->grad;
        this->m_prev[1]->grad = 1.0 * this->grad;
    }else{
        if(this->m_op == "*"){
            this->m_prev[0]->grad += this->m_prev[1]->data * this->grad;
            this->m_prev[1]->grad += this->m_prev[0]->data * this->grad;
        }else{
            if(this->m_op == "tanh"){
                this->m_prev[0]->grad = (1 - pow(this->data, 2)) * this->grad;
            }
        }
    }
}

template <typename T>
Value<T> Value<T>::operator+(Value<T> &other) {
    return Value(data + other.data, "", "+", {this, &other});
}

template <typename T>
Value<T> Value<T>::operator*(Value<T> &other){
    return Value(data * other.data, "", "*", {this, &other});
}

// Function to get the previous elements that make up this element
template <typename T> Value<T> Value<T>::tanh() {
    T x = this->data;
    T t = (exp(2 * x) - 1) / (exp(2 * x) + 1);
    return Value(t, "", "tanh", {this, nullptr});
}

template <typename T>
void Value<T>::topo_sort_helper(Value<T>* v, std::vector<Value<T>*>& sorted_values) {

    // Create a set to keep track of the visited nodes
    std::unordered_set<Value<T>*> visited;
    // Create a stack to hold the nodes
    std::stack<Value<T>*> stack;

    // Push the current node to the stack
    stack.push(v);

    // Loop until the stack is empty
    while (!stack.empty()) {
        // Get the top node of the stack
        Value<T>* current = stack.top();
        stack.pop();

        // If the current node has already been visited, continue to the next iteration
        if (visited.count(current) > 0) {
            continue;
        }

        // Mark the current node as visited and add it to the sorted values list
        visited.insert(current);
        sorted_values.emplace_back(current);

        // Iterate over the previous nodes of the current node
        for (Value<T>* child : current->m_prev) {
            // If the previous node is not null, push it to the stack
            if (child != nullptr) {
                stack.push(child);
            }
        }
    }
}

template <typename T>
void Value<T>::backprop() {
    // Create a vector to hold the sorted values
    std::vector<Value<T>*> sorted_values;

    // Perform topological sort starting from the current node
    topo_sort_helper(this, sorted_values);

    // Set the derivative of dx/dx to 1
    this->grad = 1.0;

    // Call backward in topological order applying the chain rule automatically
    for (auto value : sorted_values){
        value->m_backward();
    }
}

template <typename T>
void Value<T>::print_graph() {
    // Create a vector to hold the sorted values
    std::vector<Value<T>*> sorted_values;

    // Perform topological sort starting from the current node
    topo_sort_helper(this, sorted_values);

    for (auto value : sorted_values){
        std::cout << *value << '\n';
    }
}
