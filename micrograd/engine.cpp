//
//  engine.cpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#include "engine.hpp"

// Implementation of the constructor which initialize and empty m_prev and zero
// gradient Constructor initialize m_prev with the children of the previous
// Value and also the operator they had if they are not given uses default values
// Multiple constructors
template <typename T>
Value<T>::Value(T data, char label, std::array<Value<T>, 2> children, char op)
    : data(data), grad(0), label(label), m_prev({op, children}) {

}

template <typename T>
Value<T> Value<T>::operator+(const Value<T> &other) const {
    // Check if other is an object of type Value
    // Create a new Value object with the sum of the data values
    Value<T> result(data + other.data, { *this, other }, '+', ' ');

    /* auto backward = [](Value<T> &result) { */
    /*     result.m_op1->grad += result.grad; */
    /*     result.m_op2->grad += result.grad; */
    /* }; */

    /* result.set_backward(backward); */

    // Using move semantic to avoid unnecessary coping by value
    return std::move(result);
}
