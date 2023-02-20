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

template <typename T> class Value {
public:
    T data;
    T grad;
    char label;
public:
    // Constructor
     Value(T data, char label = ' ', char op = ' ', std::array<std::unique_ptr<Value<T>>, 2> children = {nullptr, nullptr})
         : data(data), label(label), m_op(op), m_children(std::move(children)), grad(0) {}

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

protected:
    char m_op;
    std::array<std::unique_ptr<Value<T>>, 2> m_children;
};

template <typename T>
Value<T> Value<T>::operator+(Value<T> const &other) const {
    auto result = Value(data + other.data, ' ', '+', {std::make_unique<Value>(*this), std::make_unique<Value>(other)});
    return result;
}
