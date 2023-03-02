#include <micrograd/nn.hpp>

using namespace nn;

int main() {
    /// Initialize the neural network
    auto model = MLP<double, 3>(3, {4, 4, 1});

    auto x = std::vector<std::shared_ptr<Value<double>>>{
        std::make_shared<Value<double>>(2.0),
        std::make_shared<Value<double>>(3.0),
        std::make_shared<Value<double>>(-1.0)
    };

    // auto will be an std::variant
    auto y = model(x);

    // 0 because we only have 1 neuron
    y[0]->backward();
    y[0]->draw_graph();

    for (auto &p : model.parameters()) {
        std::cout << p << "\n";
    }

    std::cout << "Model information: " << model << "\n";
    std::cout << "Number of parameters: " << model.parameters().size() << "\n";

    std::cout << "Output: " << *y[0] << "\n";
}
