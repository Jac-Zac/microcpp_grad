#include <micrograd/engine.hpp>

int main() {
    // Testing
    auto a = Value<double>(-4.0, "a");
    auto b = Value<double>(2.0, "b");
    auto c = a * 2;
    c.label = "c";
    auto d = Value<double>(0);
    d.label = "d";
    d += (b - a) ^ 2;
    d += (c - a) ^ 2;
    d.backward();
    d.draw_graph();
}
