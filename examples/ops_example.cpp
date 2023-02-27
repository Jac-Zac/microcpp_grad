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
    auto a = Value<double>(-4.0, "a");
    auto b = Value<double>(2.0, "b");
    auto c = a + b;
    c.label = "c";
    auto d = a * b + (b ^ 3);
    auto mid1 = c;
    mid1.label = "mid1";
    c = mid1 + (mid1 + 1);
    auto mid2 = c;
    mid2.label = "mid2";
    c = mid2 + ((mid2 - a) + 1);
    auto tmp1 = d;
    tmp1.label = "tmp1";
    d = tmp1 + ((tmp1 * 2) + (b + a).relu());
    auto tmp2 = d;
    tmp2.label = "tmp2";
    d = tmp2 + (3 * tmp2 + (b - a).relu());
    d.label = "d";
    auto e = c - d;
    e.label = "e";
    auto f = e ^ 2;
    f.label = "f";
    auto g = (f / 2.0);
    g.label = "g";
    auto fin = g;
    g = fin + (f.inverse_value() * 10);
    g.backward();
    g.draw_graph();
    std::cout << g << '\n';
    std::cout << a << '\n';
    std::cout << b << '\n';
}
