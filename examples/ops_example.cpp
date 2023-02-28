#include "../micrograd/engine.hpp"

int main() {
    // Testing
    auto a = Value<double>(-4.0, "a");
    auto b = Value<double>(2.0, "b");
    auto c = a + b;
    auto d = a * b + (b ^ 3);
    auto test = Value<double>(1, "test 111");
    auto mid = c;
    mid.label = "mid";
    c += (mid + test);
    c.label = "c";
    /* c += ((c - a) + 1); */
    /* auto tmp1 = d; */
    /* tmp1.label = "tmp1"; */
    /* d += ((tmp1 * 2) + (b + a).relu()); */
    /* auto tmp2 = d; */
    /* tmp2.label = "tmp2"; */
    /* d += (3 * tmp2 + (b - a).relu()); */
    /* d.label = "d"; */
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
