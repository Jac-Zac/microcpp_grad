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
    mutable T grad;    // gradient which by default is zero and it is mutable
    std::string label; // label of the value
    std::function<void()> m_backward; // lambda function
public:
    // Constructor
    Value(T data, std::string label = "", std::string op = "", std::array<Value<T>*,2> children = {});

    // Operator Overloading
    Value operator+(Value const &other) const;
    Value operator*(Value const &other) const;

    Value tanh();

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value<T> &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad
           << ", label=" << v.label << ")";
        return os;
    };

    void get_prev() const;

public:
    std::string m_op;
    std::array<Value<T>*, 2> m_prev;
};

// ==================== Implementation =====================

template <typename T>
Value<T>::Value(T data, std::string label, std::string op, std::array<Value<T>*,2> children)
    : data(data), label(label), m_op(op), m_prev(children), grad(0) {
    // Initialize the lambda to none
    m_backward = []() {};
}

template <typename T>
Value<T> Value<T>::operator+(Value<T> const &other) const {
    auto out =
        Value(data + other.data, "", "+", {const_cast<Value<T>*>(this), const_cast<Value<T>*>(&other)});
    out.m_backward = [&]() mutable {
        // Should just move the gradient along to both of them
        this->grad = 1.0 * out.grad;
        other.grad = 1.0 * out.grad;
    };
    return out;
}

template <typename T>
Value<T> Value<T>::operator*(Value<T> const &other) const {
    auto out =
        Value(data * other.data, "", "*", {static_cast<Value<T>*>(this), const_cast<Value<T>*>(&other)});
    out.m_backward = [&]() mutable {
        this->grad += other.data * out.grad;
        other.grad += this->data * out.grad;
    };
    return out;
}

// Function to get the previous elements that make up this element
template <typename T> Value<T> Value<T>::tanh() {
    T x = this->data;
    T t = (exp(2 * x) - 1) / (exp(2 * x) + 1);
    auto out = Value(t, "", "tanh", {const_cast<Value<T>*>(this), nullptr});
    out.m_backward = [this, &out, t]() mutable {
        // Chain rule with the derivative of tanh
        this->grad = (1 - pow(t, 2)) * out.grad;
    };
    return out;
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
