#include "../micrograd/engine.hpp"

int main() {

    // Creating a single perception

    // Input x1, x2
    auto x1 = Value<double>(2.0, "x1");
    auto x2 = Value<double>(0.0, "x2");
    // Weight w1, w2
    auto w1 = Value<double>(-3.0, "w1");
    auto w2 = Value<double>(1.0, "w2");

    // products
    auto x1w1 = x1 * w1;
    x1w1.label = "x1*w1";
    auto x2w2 = x2 * w2;
    x2w2.label = "x2*w2";

    // sum of the two
    auto x1w1_x2w2 = x1w1 + x2w2;
    x1w1_x2w2.label = "x1w1 + x2w2";

    // Bias of the neuron b
    auto b = Value<double>(6.881375870, "b");

    // new neuron
    auto n = x1w1_x2w2 + b;
    auto o = n.tanh();
    o.label = "o";
    n.label = "n";

    // Grandina with respect to itself is 1
    o.grad = 1.0;

    // Call backward
    o.m_backward();
    n.m_backward();
    b.m_backward();
    x1w1_x2w2.m_backward();
    x1w1.m_backward();
    x2w2.m_backward();

//    std::cout << "Children count= " << o.m_prev[0].use_count() << "\n";

//   n.data = 100;

    std::cout << (o.m_prev[0]) << "\n";
    std::cout << *(o.m_prev[0]) << "\n";
    std::cout << &n << "\n";
    std::cout << n << "\n";
    std::cout << b << "\n";
    std::cout << x1w1_x2w2 << "\n";
    std::cout << x1w1<< "\n";
    std::cout << x2w2<< "\n";
    std::cout << w2<< "\n";
    std::cout << x2<< "\n";
    std::cout << x1<< "\n";
    std::cout << w1<< "\n";

    return 0;
}
