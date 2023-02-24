//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

enum value_ops : unsigned char {
    SUM = '+',
    DIF = '-',
    MUL = '*',
    DIV = '/',
    POW = '^',
    EXP = 'e',
    NEG = 'n',
    TANH = 't',
    RELU = 'r'
};

template <typename T> class Value {
public:
    T data;            // data of the value
    T grad;            // gradient which by default is zero
    std::string label; // label of the value
public:
    // Constructor
    Value(T data, std::string label = "", char op = ' ');

    // Operator Overloading
    // lvalues version
    Value operator+(Value &other);
    Value operator*(Value &other);
    Value operator-(Value &other);
    Value operator/(Value &other);
    Value operator^(Value &other);
    // rvalues version
    Value operator+(T other);
    Value operator*(T other);
    Value operator-(T other);
    Value operator/(T other);
    Value operator^(T other);
    // Unari minus operator
    Value operator-();

    // << operator overload
    friend std::ostream &operator<<(std::ostream &os, const Value<T> &v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad
           << ", label=" << v.label << ")";
        return os;
    };

    // Other ops
    Value inverse_value();
    Value exp_value();
    Value tanh();
    // Value relu();

    void backward();
    void print_graph();

protected:
    char m_op;
    std::array<Value<T> *, 2> m_prev; // previous values
    std::vector<Value<T> *>
        m_sorted_values; // vector to store the sorted values
    std::unordered_set<Value<T> *> m_visited; // keep track of the visited nodes
protected:
    // Helper function to make a topological sort
    void _topo_sort(Value<T> *v);
    void _backward(); // 1 step of backdrop
    void _draw_graph(const Value<T>* v)const;
};

// ==================== Implementation =====================

template <typename T>
Value<T>::Value(T data, std::string label, char op)
    : data(data), label(label), m_op(op), grad(0), m_prev({nullptr, nullptr}) {}

template <typename T> Value<T> Value<T>::operator+(Value<T> &other) {
    Value<T> result = Value<T>(data + other.data, "", SUM);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T> Value<T> Value<T>::operator+(T other) {
    Value<T> result = Value<T>(data + other, "", SUM);
    result.m_prev[0] = this;
    result.m_prev[1] = new Value<T>(other, "leaf", SUM);
    return result;
}

template <typename T> Value<T> Value<T>::operator-(Value<T> &other) {
    Value<T> result = Value<T>(data - other.data, "", DIF);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T> Value<T> Value<T>::operator-(T other) {
    Value<T> result = Value<T>(data - other, "", DIF);
    result.m_prev[0] = this;
    result.m_prev[1] = new Value<T>(other, "leaf", SUM);
    return result;
}

template <typename T> Value<T> Value<T>::operator*(Value<T> &other) {
    Value<T> result = Value<T>(data * other.data, "", MUL);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T> Value<T> Value<T>::operator*(T other) {
    Value<T> result = Value<T>(data * other, "", MUL);
    result.m_prev[0] = this;
    result.m_prev[1] = new Value<T>(other, "leaf", MUL);
    return result;
}

template <typename T> Value<T> Value<T>::operator/(Value<T> &other) {
    Value<T> result = Value<T>(data / other.data, "", DIV);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T> Value<T> Value<T>::operator/(T other) {
    Value<T> result = Value<T>(data / data, "", DIV);
    result.m_prev[0] = this;
    result.m_prev[1] = new Value<T>(other, "leaf", DIV);
    return result;
}

template <typename T> Value<T> Value<T>::operator^(Value<T> &other) {
    Value<T> result = Value<T>(pow(data, other.data), "", POW);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T> Value<T> Value<T>::operator^(T other) {
    Value<T> result = Value<T>(pow(data, other), "", POW);
    result.m_prev[0] = this;
    result.m_prev[1] = new Value<T>(other, "leaf", POW);
    return result;
}

template <typename T> Value<T> Value<T>::operator-() { return *this * (-1); }

template <typename T> Value<T> Value<T>::inverse_value() {
    return *this ^ (-1);
}

template <typename T> Value<T> Value<T>::exp_value() {
    Value<T> result = Value<T>(exp(data), "", EXP);
    result.m_prev[0] = this;
    result.m_prev[1] = nullptr;
    return result;
}

template <typename T> Value<T> Value<T>::tanh() {
    T x = this->data;
    T tanh = (exp(2 * x) - 1) / (exp(2 * x) + 1);
    Value<T> result = Value<T>(tanh, "", TANH);
    result.m_prev[0] = this;
    result.m_prev[1] = nullptr;
    return result;
}

template <typename T> void Value<T>::_backward() {
    switch (this->m_op) {
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
        this->m_prev[0]->grad += (1 / (this->m_prev[1]->data)) * this->grad;
        this->m_prev[1]->grad -= (this->m_prev[0]->data) /
                                 pow(this->m_prev[1]->data, 2) * this->grad;
        break;
    case POW:
        this->m_prev[0]->grad +=
            (this->m_prev[1]->data *
             pow(this->m_prev[0]->data, (this->m_prev[1]->data - 1))) *
            this->grad;
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

template <typename T> void Value<T>::_topo_sort(Value<T> *v) {
    // Add it to the visited values
    m_visited.insert(v);
    // Iterate trough the children
    for (auto *child : v->m_prev) {
        // If not visited and not a leaf value
        if (m_visited.count(child) == 0 && child != nullptr) {
            // Call the function recursively
            _topo_sort(child);
        }
    }
    m_sorted_values.push_back(v);
}

template <typename T> void Value<T>::backward() {

    // If empty do topo sort
    if (m_sorted_values.empty()) {
        _topo_sort(this);
        std::reverse(m_sorted_values.begin(), m_sorted_values.end());
    }

    // Set the derivative of dx/dx to 1
    this->grad = 1.0;

    // Call backward in topological order applying the chain rule automatically
    for (auto &value : m_sorted_values) {
        value->_backward();
    }
}

template <typename T> void Value<T>::print_graph() {
    // If empty do topo sort
    if (m_sorted_values.empty()) {
        _topo_sort(this);
        std::reverse(m_sorted_values.begin(), m_sorted_values.end());
    }

#define ASCII_DRAWING 1
#if ASCII_DRAWING == 1
    for (auto &value : m_sorted_values){
        std::cout << *value;
        if (value->m_op != ' '){
            std::cout << '\n' << std::string(20, ' ') << "|" << '\n' << std::string(20, ' ') << value->m_op << std::string(10, ' ') << '\n' << std::string(20, ' ') << "|" << '\n';
        }else{
            std::cout << '\n';
        }
    }
#else
    for (auto &value : m_sorted_values) {
        std::cout << *value << '\n';
    }
#endif
}
