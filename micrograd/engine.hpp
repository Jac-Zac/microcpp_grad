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
    std::string label; // label of the value
public:
    // Constructor
    Value(T data, std::string label = "", char op = ' ');
    ~Value();

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
        os << "Value(data=" << v.data << ", grad=" << v.m_grad
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
    T data; // data of the value
    char m_op;
    T m_grad;                         // gradient which by default is zero
    std::array<Value<T> *, 2> m_prev; // previous values
    std::vector<Value<T> *>
        m_sorted_values; // vector to store the sorted values
    std::unordered_set<Value<T> *> m_visited; // keep track of the visited nodes
protected:
    // Helper function to make a topological sort
    void _topo_sort(Value<T> *v);
    void _backward(); // 1 step of backdrop
};

// ==================== Implementation =====================

template <typename T>
Value<T>::Value(T data, std::string label, char op)
    : data(data), label(label), m_op(op), m_grad(0),
      m_prev({nullptr, nullptr}) {}

template <typename T>
Value<T>::~Value(){

}

template <typename T> Value<T> Value<T>::operator+(Value<T> &other) {
    Value<T> result = Value<T>(data + other.data, "", SUM);
    result.m_prev[0] = this;
    result.m_prev[1] = &other;
    return result;
}

template <typename T> Value<T> Value<T>::operator+(T other) {
    Value<T> result = Value<T>(data + other, "", SUM);
    result.m_prev[0] = this;
    result.m_prev[1] = new Value<T>(other, "leaf");
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
    result.m_prev[1] = new Value<T>(other, "leaf");
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
    result.m_prev[1] = new Value<T>(other, "leaf");
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
    result.m_prev[1] = new Value<T>(other, "leaf");
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
    result.m_prev[1] = new Value<T>(other, "leaf");
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

template <typename T> Value<T> Value<T>::relu() {
    T relu = this->data < 0 ? 0 : this->data;
    Value<T> result = Value<T>(relu, "", RELU);
    result.m_prev[0] = this;
    result.m_prev[1] = nullptr;
    return result;
}

template <typename T> void Value<T>::_backward() {
    switch (this->m_op) {
    case SUM:
        // Should just move the gradient along to both of them
        // += because we want to avoid bugs if we reuse a variable
        this->m_prev[0]->m_grad += 1.0 * this->m_grad;
        this->m_prev[1]->m_grad += 1.0 * this->m_grad;
        break;
    case DIF:
        this->m_prev[0]->m_grad += 1.0 * this->m_grad;
        this->m_prev[1]->m_grad -= 1.0 * this->m_grad;
        break;
    case MUL:
        this->m_prev[0]->m_grad += this->m_prev[1]->data * this->m_grad;
        this->m_prev[1]->m_grad += this->m_prev[0]->data * this->m_grad;
        break;
    case DIV:
        this->m_prev[0]->m_grad += (1 / (this->m_prev[1]->data)) * this->m_grad;
        this->m_prev[1]->m_grad -= (this->m_prev[0]->data) /
                                   pow(this->m_prev[1]->data, 2) * this->m_grad;
        break;
    case POW:
        this->m_prev[0]->m_grad +=
            (this->m_prev[1]->data *
             pow(this->m_prev[0]->data, (this->m_prev[1]->data - 1))) *
            this->m_grad;
        break;
    case EXP:
        // e^x is e^x which I already saved in this->data
        this->m_prev[0]->m_grad += this->data * this->m_grad;
        break;
    case TANH:
        this->m_prev[0]->m_grad += (1 - pow(this->data, 2)) * this->m_grad;
        break;
    case RELU:
        this->m_prev[0]->m_grad += this->data > 0 ? this->data : 0;
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
    this->m_grad = 1.0;

    // Call backward in topological order applying the chain rule automatically
    for (auto &value : m_sorted_values) {
        value->_backward();
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
                << " | data = " << values->data
                << " | grad = " << values->m_grad << "\", shape=record]\n";
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
    std::system("dot -Tpng graph.dot -o graph.png");
    // Open the graph using the default viewer
    std::system("open graph.png");
}
