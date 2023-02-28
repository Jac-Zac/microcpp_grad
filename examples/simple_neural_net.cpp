#include "../micrograd/nn.hpp"

int main() {
    /// Initialize the neural network
    auto n = MLP<double, 3> (3, {4,4,1});

    std::vector<Value<TYPE>> x = {
        Value<double>(2.0, "first_value"),
        Value<double>(3.0, "second_value"),
        Value<double>(-1.0, "third_value"),
    };

    // auto will be an std::variant
    auto y = n(x);

    std::get<Value<double>>(y).backward();
    std::get<Value<double>>(y).draw_graph();
}
