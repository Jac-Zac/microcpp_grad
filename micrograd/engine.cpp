//
//  engine.cpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#include "engine.hpp"

template <typename T>
Value<T>::Value(T data, std::array<Value<T>,2> children, char _op=' ') : data(data), grad(0)
{

    // Variables to autograph the contractor
    // self.m_bacward =
    // self.prev = (children)
}

template <typename T>
Value<T> Value<T>::operator+(const Value<T> &other) const {
    T result = this->data + other.data;

    /* auto backward = [](Value<T> &result) { */
    /*     result.m_op1->grad += result.grad; */
    /*     result.m_op2->grad += result.grad; */
    /* }; */

    result
    /* result.set_backward(backward); */

    return result;
}
