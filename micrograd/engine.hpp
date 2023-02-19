//
//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.
//

#include<iostream>
#include<string>
#include<array>
#include<set>

#pragma once

template <typename T>
class Value;

// Create a template for previous elements
template<typename T>
struct Previous{
    char _op = ' ';
    std::array<Value<T>,2> children;
};

template <typename T> class Value {
public:
    T data;
    T grad;
public:
    // Constructor
    Value(T data, std::array<Value<T>,2> children, char op);
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
    Previous<T> _prev;
};
