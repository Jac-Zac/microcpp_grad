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
#include <iostream>

template <typename T> class Value {
public:
    T data; // data of the value
    T grad; //gradient which by default is zero
    std::string label; // label of the value
public:
    // Constructor
    Value(T data, std::string label = "", std::string op = "",
          std::array<std::shared_ptr<Value<T>>, 2> children = {nullptr,
                                                               nullptr})
        : data(data), label(label), m_op(op), m_prev(std::move(children)),
          grad(0) {}

    // Operator Overloading
    Value operator+(Value const &obj) const;
    Value operator*(Value const &obj) const;

    Value tanh();

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
        return os;
    };

    void get_prev();

protected:
    std::string m_op;
    std::array<std::shared_ptr<Value<T>>, 2> m_prev;
};

template <typename T>
Value<T> Value<T>::operator+(Value<T> const &other) const {
    auto result = Value(data + other.data, "", "+",
                        {std::make_shared<Value>(std::move(*this)),
                         std::make_shared<Value>(std::move(other))});
    return result;
}

template <typename T>
Value<T> Value<T>::operator*(Value<T> const &other) const {
    auto result = Value(data * other.data, "", "*",
                        {std::make_shared<Value>(std::move(*this)),
                         std::make_shared<Value>(std::move(other))});
    return result;
}

// Function to get the previous elements that make up this element
template <typename T>
Value<T> Value<T>::tanh(){
    T x = this->data;
    T t = (exp(2 * x) - 1)/(exp(2 * x) + 1);
    auto result = Value(t, "", "tanh", {std::make_shared<Value>(std::move(*this)), nullptr});
    return result;
}

// Function to get the previous elements that make up this element
template <typename T>
void Value<T>::get_prev(){
    if (this->m_prev[1] != nullptr){
        std::cout << "{" << *(this->m_prev[0]) << "," << *(this->m_prev[1]) << "}\n";
    }else{
        std::cout << "{" << *(this->m_prev[0]) << "}\n";
    }
}
