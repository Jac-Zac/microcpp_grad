#include "../micrograd/engine.hpp"
#include <__tuple>

int main() {
    Value<double> a = Value<double>(0.8814, "a");
    Value<double> b = Value<double>(3.0, "b");
    auto o = a.tanh();
    o.label = "o";
    o.grad = 1.0;
    a.m_backward();
    std::cout << a << "," << std::endl;
    std::cout << o << "," << std::endl;
    o.get_prev();

    return 0;
}
