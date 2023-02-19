//
//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#include<iostream>
#include<array>

#pragma once

template <typename T> class Value;

// Create a template for previous elements
template<typename T>
struct Previous{
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
    Value(T data, char label = ' ', char op = ' ', std::array<std::unique_ptr<Value<T>>, 2> children = {});

    // Operator Overloading
    Value operator+(Value const &obj) const;
    Value operator-(Value const &obj) const;
    Value operator*(Value const &obj) const;
    Value operator/(Value const &obj) const;

    // << operator overload
    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
        return os;
    };

private:
    Previous<T> m_prev;
};
