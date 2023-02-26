#include "../micrograd/nn.hpp"

#define INPUTS 3

typedef double TYPE;

int main() {
    // Binary classification

    // define the neural network
    auto n = MLP<TYPE, INPUTS>(INPUTS, {4, 4, 1});

    // std::cout << n; // to output the network shape

    std::vector<std::vector<Value<TYPE>>> xs = {{2.0, 3.0, -1.0},
                                                {3.0, -1.0, 0.5},
                                                {0.5, 1.0, 1.0},
                                                {1.0, 1.0, -1.0}};

    // desired target
    std::vector<Value<TYPE>> ys = {1.0, -1.0, -1.0, 1.0};

    std::vector<Value<TYPE>> ypred;

    // Forward pass
    // Run the 4 different example through the network
    for(size_t i = 0; i < 4 ; i++){
        ypred.emplace_back(n(xs[0])[i]);
    }


    for (auto &value : ypred) {
        /* std::cout << value << '\n'; */
        value.draw_graph();
    }


    Value<TYPE> loss = Value<TYPE>(0, "loss");

    Value<TYPE> tmp = Value<TYPE>(0, "tmp");

    for (size_t i = 0; i < 4; i++){
        // Mean Squared Error
        tmp = (ypred[i] - ys[i]);
        loss = loss + (tmp ^ 2);
    }

    std::cout << n.m_layers[0].m_neurons[0].m_weights[0] << '\n';

    loss.backward();

    std::cout << loss << '\n';

    loss.draw_graph();

    std::cout << n.m_layers[0].m_neurons[0].m_weights[0] << '\n';

}
