//
//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#pragma once

#include <array>
#include <iostream>
#include <memory>

// Forward declaration
template <typename T> class Value;

// Create a template for previous elements
template <typename T> struct Previous {
    char m_op;
    std::array<std::unique_ptr<Value<T>>, 2> m_children;
};

template <typename T> class Value {
public:
    T data;
    T grad;
    char label;

public:
    // Constructor
    Value(T data, char label = ' ', char op = ' ',
          std::array<std::unique_ptr<Value<T>>, 2> children = {});

    // Operator Overloading
    Value operator+(Value const &obj) const;
    Value operator-(Value const &obj) const;
    Value operator*(Value const &obj) const;
    Value operator/(Value const &obj) const;

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
        return os;
    };

private:
    Previous<T> m_prev;
};

template <typename T>
inline Value<T>::Value(T data, char label, char op,
                std::array<std::unique_ptr<Value<T>>, 2> children)
    : data(data), label(label), m_prev({op, children}), grad(0) {
    // self.m_bacward =
}

template <typename T>
inline Value<T> Value<T>::operator+(Value<T> const &other) const {
    Value<T> result = Value(
        data + other.data, ' ', '+',
        {std::make_unique<Value<T>>(*this), std::make_unique<Value<T>>(other)});
    /*
    auto backward = [](Value<T> &result) {
        result.m_op1->grad += result.grad;
        result.m_op2->grad += result.grad;
    };
    result.set_backward(backward);
    */

    // Using move semantic to avoid unnecessary coping by value
    return std::move(result);
}
