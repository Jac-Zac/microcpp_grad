//  engine.hpp
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

enum value_ops : char {
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
    std::string label; // label of the value
    T data;            // data of the value
    T grad;            // gradient which by default is zero
public:
    // Constructor
    Value(T data, std::string label = "", char op = ' ');

    // Operator Overloading
    // lvalues and rvalues because of const reference
    Value operator+(const Value &other) const;
    Value operator*(const Value &other) const;
    Value operator-(const Value &other) const;
    Value operator/(const Value &other) const;
    Value operator^(const Value &other) const;

    // Unari minus operator
    Value operator-() const;

    Value &operator+=(const Value &other);
    Value &operator-=(const Value &other);
    Value &operator*=(const Value &other);
    Value &operator/=(const Value &other);

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
    Value relu();

    void backward();
    void draw_graph();

protected:
    char m_op;
    std::array<Value<T> *, 2> m_prev; // previous values
    std::vector<Value<T> *>
        m_sorted_values; // vector to store the sorted values
    std::unordered_set<Value<T> *> m_visited; // keep track of the visited nodes
protected:
    // Helper function to make a topological sort
    void _topo_sort(Value<T> *v);
    void _backward_single(); // 1 step of backdrop
    // Function to update the gradient
    void _update_grad(T grad) { this->grad += grad; }
};

// ==================== Implementation =====================

template <typename T>
Value<T>::Value(T data, std::string label, char op)
    : data(data), label(label), m_op(op), grad(0.0),
      m_prev({nullptr, nullptr}) {}

template <typename T> Value<T> &Value<T>::operator+=(const Value<T> &other) {
    // update the data
    data += other.data;
    m_prev[0] = const_cast<Value<T> *>(this);
    m_prev[1] = const_cast<Value<T> *>(&other);
    return *this;
}

template <typename T> Value<T> &Value<T>::operator-=(const Value<T> &other) {
    data -= other.data;
    m_prev[0] = const_cast<Value<T> *>(this);
    m_prev[1] = const_cast<Value<T> *>(&other);
    return *this;
}

template <typename T> Value<T> &Value<T>::operator*=(const Value<T> &other) {
    data *= other.data;
    m_prev[0] = const_cast<Value<T> *>(this);
    m_prev[1] = const_cast<Value<T> *>(&other);
    return *this;
}

template <typename T> Value<T> &Value<T>::operator/=(const Value<T> &other) {
    data /= other.data;
    m_prev[0] = const_cast<Value<T> *>(this);
    m_prev[1] = const_cast<Value<T> *>(&other);
    return *this;
}

template <typename T>
Value<T> Value<T>::operator+(const Value<T> &other) const {
    Value<T> result = Value<T>(data + other.data, "", SUM);
    result.m_prev[0] = const_cast<Value<T> *>(this);
    result.m_prev[1] = const_cast<Value<T> *>(&other);
    return result;
}

template <typename T>
Value<T> Value<T>::operator-(const Value<T> &other) const {
    Value<T> result = Value<T>(data - other.data, "", DIF);
    result.m_prev[0] = const_cast<Value<T> *>(this);
    result.m_prev[1] = const_cast<Value<T> *>(&other);
    return result;
}

template <typename T>
Value<T> Value<T>::operator*(const Value<T> &other) const {
    Value<T> result = Value<T>(data * other.data, "", MUL);
    result.m_prev[0] = const_cast<Value<T> *>(this);
    result.m_prev[1] = const_cast<Value<T> *>(&other);
    return result;
}

template <typename T>
Value<T> Value<T>::operator^(const Value<T> &other) const {
    Value<T> result = Value<T>(pow(data, other.data), "", POW);
    result.m_prev[0] = const_cast<Value<T> *>(this);
    result.m_prev[1] = const_cast<Value<T> *>(&other);
    return result;
}

template <typename T>
Value<T> Value<T>::operator/(const Value<T> &other) const {
    Value<T> result = Value<T>(data / other.data, "", DIV);
    result.m_prev[0] = const_cast<Value<T> *>(this);
    result.m_prev[1] = const_cast<Value<T> *>(&other);
    return result;
}

template <typename T> Value<T> Value<T>::operator-() const {
    return *this * (-1);
}

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
    Value<T> result = Value<T>(std::tanh(data), "", TANH);
    result.m_prev[0] = this;
    result.m_prev[1] = nullptr;
    return result;
}

template <typename T> Value<T> Value<T>::relu() {
    T relu = data < 0 ? 0 : data;
    Value<T> result = Value<T>(relu, "", RELU);
    result.m_prev[0] = this;
    result.m_prev[1] = nullptr;
    return result;
}

template <typename T> void Value<T>::_backward_single() {
    switch (this->m_op) {
    case SUM:
        // Should just move the gradient along to both of them
        // += because we want to avoid bugs if we reuse a variable
        m_prev[0]->_update_grad(grad);
        m_prev[1]->_update_grad(grad);
        break;
    case DIF:
        // same as m_prev[0] += 1.0 * grad;
        m_prev[0]->_update_grad(grad);
        m_prev[1]->_update_grad(-grad); // same as doing -=
        break;
    case MUL:
        // same as m_prev[0] += m_prev[1]->data * grad
        m_prev[0]->_update_grad(m_prev[1]->data * grad);
        m_prev[1]->_update_grad(m_prev[0]->data * grad);
        break;
    case DIV:
        m_prev[0]->_update_grad((1 / (m_prev[1]->data)) * grad);
        m_prev[1]->_update_grad(-(m_prev[0]->data) / pow(m_prev[1]->data, 2) *
                                grad);
        break;
    case POW:
        m_prev[0]->_update_grad(
            (m_prev[1]->data * pow(m_prev[0]->data, (m_prev[1]->data - 1))) *
            grad);
        break;
    case EXP:
        // e^x is e^x which I already saved in data
        m_prev[0]->_update_grad(data * grad);
        break;
    case TANH:
        m_prev[0]->_update_grad((1 - pow(data, 2)) * grad);
        break;
    case RELU:
        m_prev[0]->_update_grad(data > 0 ? 1 : 0);
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
    }

    // Set the derivative of dx/dx to 1
    this->grad = 1.0;

    // Call backward in topological order applying the chain rule automatically
    for (auto it = m_sorted_values.rbegin(); it != m_sorted_values.rend();
         ++it) {
        (*it)->_backward_single();
    }
}

template <typename T> void Value<T>::draw_graph() {
    // Perform a topological sort of the graph in reverse
    if (m_sorted_values.empty()) {
        _topo_sort(this);
    }

    // Open a file to write the output
    std::ofstream outfile("graph.dot");
    if (!outfile) {
        std::cerr << "Error: failed to open graph.dot\n";
        return;
    }

    // Create a graphviz graph
    outfile << "digraph G {\n";
    outfile << "  rankdir=LR; // set rankdir attribute to LR\n";
    for (const auto &values : m_sorted_values) {
        outfile << "  " << uintptr_t(values)
                << " [label=\"label = " << values->label
                << " | data = " << values->data << " | grad = " << values->grad
                << "\", shape=record]\n";
        if (values->m_op != ' ') {
            // if this value is a result of some operation, create an op node
            // for it
            outfile << "  " << uintptr_t(values) + values->m_op << " [label=\""
                    << values->m_op << "\"]\n";
            outfile << "  " << uintptr_t(values) + values->m_op << " -> "
                    << uintptr_t(values) << "\n";
        }
        for (size_t j = 0; j < 2; j++)
            if (values->m_prev[j] != nullptr) {
                // if this value is a result of some operation, create an op
                // node for it
                outfile << "  " << uintptr_t(values->m_prev[j]) << " -> "
                        << uintptr_t(values) + values->m_op << "\n";
            }
    }

    outfile << "}\n";
    outfile.close();

    // Create the graph using the dot command
    std::system("dot -Tpng graph.dot -Gdpi=300 -o graph.png");
    // Open the graph using the default viewer
    std::system("open graph.png");
}
