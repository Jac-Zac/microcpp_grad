/* #include "../micrograd/engine.hpp" */
/*  */
/* int main() { */
/*     // Creating a single perception */
/*  */
/*     // Input x1, x2 */
/*     auto x1 = Value<double>(2.0, "x1"), x2 = Value<double>(0.0, "x2"); */
/*     // Weight w1, w2 */
/*     auto w1 = Value<double>(-3.0, "w1"), w2 = Value<double>(1.0, "w2"); */
/*  */
/*     // products */
/*     auto x1w1 = x1 * w1; */
/*     x1w1.label = "x1*w1"; */
/*     auto x2w2 = x2 * w2; */
/*     x2w2.label = "x2*w2"; */
/*  */
/*     // sum of the two */
/*     auto x1w1_x2w2 = x1w1 + x2w2; */
/*     x1w1_x2w2.label = "x1w1 + x2w2"; */
/*  */
/*     // Bias of the neuron b */
/*     auto b = Value<double>(6.8813735870195432, "b"); */
/*  */
/*     // new neuron */
/*     auto n = (x1w1_x2w2 + b); */
/*     n.label = "n"; */
/*  */
/*     // auto o = n.tanh(); */
/*  */
/*     // Custom tanh implementation */
/*     auto e = (n * 2).exp_value(); */
/*     e.label = "e"; */
/*     auto o = (e - 1) / (e + 1); */
/*     o.label = "o"; */
/*  */
/*     // Grandina with respect to itself is 1 */
/*     o.backward(); */
/*     o.draw_graph(); */
/* } */

#include "../micrograd/engine.hpp"

int main() {
    // Testing
    auto a = Value<double>(-4.0);
    auto b = Value<double>(2.0);
    auto c = a + b;
    auto d = a * b + (b ^ 3);
    c = c + c + 1;
    c = (c + 1 + c) - a;
    d = d + d * 2 + (b + a).relu();
    d = d + (d * 3) + (b - a).relu();
    auto e = c - d;
    auto f = e * e;
    auto g = (f / 2.0);
    g.label = "g";
    auto mid = (f.inverse_value() * 10);
    g += mid;
    g.backward();
    g.draw_graph();
    std::cout << g << '\n';
    std::cout << a << '\n';
    std::cout << b << '\n';
}
