#include <micrograd/nn.hpp>

using namespace nn;

/* #define TEST */
#ifdef TEST

int main() {
    /// Initialize the neural network
    auto n = MLP<double, 3> (3, {4,4,1});

    std::vector<Value<double>> x = {
        Value<double>(2.0, "first_value"),
        Value<double>(3.0, "second_value"),
        Value<double>(-1.0, "third_value"),
    };

    // auto will be an std::variant
    auto y = n(x);

    y[3][0].backward();
    y[3][0].draw_graph();
}

#else

int main() {
    /// Initialize the neural network
    auto model = MLP<double, 3>(3, {4, 4, 1});

    std::vector<Value<double>> x = {
        Value<double>(2.0, "first_value"),
        Value<double>(3.0, "second_value"),
        Value<double>(-1.0, "third_value"),
    };

    // auto will be an std::variant
    auto y = model(x);

    // 0 because we only have 1 neuron
    y[3][0].backward();
    model.zero_grad();

    for (auto &p : model.parameters()) {
        std::cout << *p << "\n";
    }

    std::cout << "Model information: " << model << "\n";
    std::cout << "Number of parameters: " << model.parameters().size() << "\n";

    std::cout << "Output: " << y[3][0] << "\n";

    y[3][0].draw_graph();

    return 0;
}

#endif
