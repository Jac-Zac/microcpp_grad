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
#include <unordered_set>

enum value_ops : unsigned char
{
    SUM = '+', DIF = '-', MUL = '*', DIV = '/', POW = '^' , EXP = 'e', NEG = 'n', TANH = 't', RELU = 'r'
};

template <typename T> class Value {
public:
    T data;             // data of the value
    T grad;             // gradient which by default is zero
    std::string label;  // label of the value
public:
    // Constructor
    Value(T data, std::string label = "", char op = ' ');

    // Operator Overloading
    Value operator+(Value &other);
    Value operator+(T other);
    Value operator*(Value &other);
    Value operator*(T other);
    Value operator-(Value &other);
    Value operator-(T other);
    Value operator/(Value &other);
    Value operator/(T other);
    // pow
    Value operator^(Value &other);
    Value operator^(T other);

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value<T> &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad
           << ", label=" << v.label << ")";
        return os;
    };

    // Other ops
    Value neg_value();
    Value exp_value();
    Value tanh();

    // Value relu();

    void backword();
    void print_graph();

public:
    char m_op;
    std::array<Value<T> *, 2> m_prev; // previous values
protected:
    // Helper function to make a topological sort
    void topo_sort_helper(Value<T>* v, std::vector<Value<T>*>& sorted_values);
    void m_backward(); // 1 step of backdrop
};

// ==================== Implementation =====================

template <typename T>
Value<T>::Value(T data, std::string label, char op)
    : data(data), label(label), m_op(op), grad(0), m_prev({nullptr,nullptr}) {}

template <typename T>
void Value<T>::m_backward(){
    switch(this->m_op){
        case SUM:
            // Should just move the gradient along to both of them
            // += because we want to avoid bugs if we reuse a variable
            this->m_prev[0]->grad += 1.0 * this->grad;
            this->m_prev[1]->grad += 1.0 * this->grad;
            break;
        case DIF:
            this->m_prev[0]->grad += 1.0 * this->grad;
            this->m_prev[1]->grad -= 1.0 * this->grad;
            break;
        case MUL:
            this->m_prev[0]->grad += this->m_prev[1]->data * this->grad;
            this->m_prev[1]->grad += this->m_prev[0]->data * this->grad;
            break;
        case DIV:
            // think about it
            // this->m_prev[0]->grad += (this->m_prev[1]->data)/(this->grad;
            // this->m_prev[1]->grad += this->m_prev[0]->data * this->grad;
            break;
        case POW:
            this->m_prev[0]->grad += (this->m_prev[1]->data * pow(this->m_prev[0]->data,(this->m_prev[1]->data - 1))) * this->grad;
            break;
        case EXP:
            // e^x is e^x which I already saved in this->data
            this->m_prev[0]->grad += this->data * this->grad;
            break;
        case TANH:
            this->m_prev[0]->grad += (1 - pow(this->data, 2)) * this->grad;
            break;
        case RELU:
            // this->m_prev[0]->grad += (1 - pow(this->data, 2)) * this->grad;
            break;
        default:
            break;
    }
}

template <typename T>
Value<T> Value<T>::operator+(Value<T> &other) {
    Value<T> result = Value<T>(data + other.data, "", SUM);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T>
Value<T> Value<T>::operator+(T other){
    Value<T> result = Value<T>(data + other, "", SUM);
    result.m_prev[0] = this;
    return result;
}

template <typename T>
Value<T> Value<T>::operator-(Value<T> &other){
    Value<T> result = Value<T>(data - other.data, "", DIF);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T>
Value<T> Value<T>::operator-(T other){
    Value<T> result = Value<T>(data - other, "", DIF);
    result.m_prev[0] = this;
    return result;
}

template <typename T>
Value<T> Value<T>::operator*(Value<T> &other){
    Value<T> result = Value<T>(data * other.data, "", MUL);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T>
Value<T> Value<T>::operator*(T other){
    Value<T> result = Value<T>(data * other, "", MUL);
    result.m_prev[0] = this;
    return result;
}

template <typename T>
Value<T> Value<T>::operator/(Value<T> &other){
    Value<T> result = Value<T>(data / other.data, "", DIV);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T>
Value<T> Value<T>::operator/(T other){
    Value<T> result = Value<T>(data / other, "", DIV);
    result.m_prev[0] = this;
    return result;
}

template <typename T>
Value<T> Value<T>::operator^(Value<T> &other){
    Value<T> result = Value<T>(pow(data,other.data), "", POW);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

/*
template <typename T> Value<T> Value<T>::neg_value() {
    Value<T> result = *this * -1;
    result.label = NEG;
    result.m_prev[0] = this;
    return result;
}
*/

template <typename T> Value<T> Value<T>::exp_value() {
    Value<T> result = Value<T>(exp(data),"",EXP);
    result.m_prev[0] = this;
    return result;
}

template <typename T> Value<T> Value<T>::tanh() {
    T x = this->data;
    T tanh = (exp(2 * x) - 1) / (exp(2 * x) + 1);
    Value<T> result = Value<T>(tanh , "", TANH);
    result.m_prev[0] = this;
    return result;
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
void Value<T>::backword() {
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
