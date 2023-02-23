#include "../micrograd/engine.hpp"

int main() {
    // Creating a single perception

    // Input x1, x2
    auto x1 = Value<double>(2.0, "x1"), x2 = Value<double>(0.0, "x2");
    // Weight w1, w2
    auto w1 = Value<double>(-3.0, "w1"), w2 = Value<double>(1.0, "w2");

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
    n.label = "n";

    // auto o = n.tanh();

    auto n2 = n*2;
    n2.label = "2*n";
    auto e = (n2).exp_value();
    e.label = "e";

    auto mid1 = (e - 1);
    mid1.label = "mid1";
    auto mid2 = (e + 1);
    mid2.label = "mid2";
    auto o = mid1/mid2;

    o.label = "o";

    /* o.grad = 1.0 */
    /* o.m_backward(); */
    /* mid1.m_backward(); */
    /* mid2.m_backward(); */
    /* e.m_backward(); */
    /* n2.m_backward(); */
    /* n.m_backward(); */
    /* x1w1_x2w2.m_backward(); */
    /* x1w1.m_backward(); */
    /* x2w2.m_backward(); */
    /* x1.m_backward(); */
    /* x2.m_backward(); */
    /* w1.m_backward(); */
    /* w2.m_backward(); */

    // Grandina with respect to itself is 1
    o.backward();
    o.print_graph();
}
