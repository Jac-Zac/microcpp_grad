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
#include <functional>
#include <iostream>
#include <memory>
#include <string>

template <typename T> class Value {
public:
    T data;            // data of the value
    T grad;    // gradient which by default is zero
    std::string label; // label of the value
public:
    // Constructor
    Value(T data, std::string label = "", std::string op = "",
          std::array<Value<T> *, 2> children = {});

    // Operator Overloading
    Value operator+(Value &other);
    Value operator*(Value &other);

    Value tanh();

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value<T> &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad
           << ", label=" << v.label << ")";
        return os;
    };

    void get_prev() const;
private:
    std::string m_op;
    std::array<Value<T> *, 2> m_prev; // previous values
public:
    void m_backward(); // lambda function
};

// ==================== Implementation =====================

template <typename T>
Value<T>::Value(T data, std::string label, std::string op,
                std::array<Value<T> *, 2> children)
    : data(data), label(label), m_op(op), m_prev(children), grad(0) {}

template <typename T>
void Value<T>::m_backward(){
    switch(this->label){
        case "+":
            // Should just move the gradient along to both of them
            this->m_prev[0]->grad = 1.0 * this->grad;
            this->m_prev[1]->grad = 1.0 * this->grad;
            break;
        case "*":
            this->m_prev[0]->grad += this->m_prev[1]->data * this->grad;
            this->m_prev[1]->grad += this->m_prev[0]->data * this->grad;
            /* break; */
        case "tanh":
            this->m_prev[0]->grad = (1 - pow(this->data, 2)) * this->grad;
            break;
        default:
            break;
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

// Function to get the previous elements that make up this element
template <typename T> void Value<T>::get_prev() const {
    if (this->m_prev[1] != nullptr) {
        std::cout << "{" << *(this->m_prev[0]) << "," << *(this->m_prev[1])
                  << "}\n";
    } else {
        std::cout << "{" << *(this->m_prev[0]) << "}\n";
    }
}
