#include <micrograd/nn.hpp>

#define INPUTS 3

typedef double TYPE;

#define LAYER

#ifdef LAYER

int main() {
    // Three neurons
    auto layer = Layer<TYPE>(3,3);

    std::vector<Value<TYPE>> x1 = {
        Value<TYPE>(-5.0, "first_value"),
        Value<TYPE>(3.0, "second_value"),
        Value<TYPE>(7.0, "third_value"),
    };

    std::vector<Value<TYPE>> x2 = {
        Value<TYPE>(3.0, "first_value"),
        Value<TYPE>(-3.0, "second_value"),
        Value<TYPE>(2.0, "third_value"),
    };

    // Testing the neuron output with two different set of values
    auto y1 = layer(x1);
    for (auto& value : y1){
        value->backward();
    }

    layer.zero_grad();

    auto y2 = layer(x2);
    for (auto& value : y2){
        value->backward();
    }

    std::cout << "Outputs:" << '\n';
    std::cout << "-----------------" << '\n';

    std::cout << "First pass: " << *y1[0] << '\n';
    /* // Neuron two should have rest the m_neurons value */
    std::cout << "Second pass: " << *y2[1] << "\n";

    std::cout << "Parameters: " << '\n';
    std::cout << "-----------------" << '\n';

    // Getting the neuron parameters
    for (auto &p : layer.parameters()) {
        std::cout << *p << "\n";
    }

    y2[0]->draw_graph();
}

#elif NETWORK

int main() {
    /// Initialize the neural network
    auto model = MLP<TYPE, INPUTS>(3, {4, 4, 1});

    std::vector<Value<TYPE>> x = {
        Value<TYPE>(2.0, "first_value"),
        Value<TYPE>(3.0, "second_value"),
        Value<TYPE>(-1.0, "third_value"),
    };

    for (auto &p : model.parameters()) {
        std::cout << p << "\n";
    }

    std::cout << "Number of parameters: " << model.parameters().size() << "\n";

    // auto will be an std::variant
    /* auto y = model(x); */
    /*  */
    /* std::get<Value<TYPE>>(y).backward(); */
    /* std::get<Value<TYPE>>(y).draw_graph(); */
}

int main() {
    // Binary classification

    // define the neural network
    auto n = MLP<TYPE, INPUTS>(INPUTS, {4, 4, 1});

    // std::cout << n; // to output the network shape

    std::vector<std::vector<Value<TYPE>>> xs = {
        {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};

    // desired target
    std::vector<Value<TYPE>> ys = {1.0, -1.0, -1.0, 1.0};

    std::vector<Value<TYPE>> ypred;

    // Forward pass
    // Run the 4 different example through the network
    for (size_t i = 0; i < 4; i++) {
        ypred.emplace_back(n(xs[0])[i]);
    }

    /* for (auto &value : ypred) { */
    /*     std::cout << value << '\n'; */
    /* } */

    auto loss = Value<TYPE>(0, "loss");

    for (size_t i = 0; i < 4; i++) {
        // Mean Squared Error
        loss += (ypred[i] - ys[i]) ^ 2;
    }

    std::cout << n.m_layers[0].m_neurons[0].m_weights[0] << '\n';

    loss.backward();

    std::cout << loss << '\n';

    std::cout << n.m_layers[0].m_neurons[0].m_weights[0] << '\n';

    loss.draw_graph();
}

#endif
