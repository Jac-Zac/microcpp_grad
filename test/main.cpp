#include <micrograd/nn.hpp>

using namespace nn;

#define SIZE 3
#define PASS_NUM 4

typedef double TYPE;

int main() {
    // Binary classification

    // define the neural network
    std::array<size_t, SIZE> n_neurons_for_layer = {4, 4, 1};
    auto model = MLP<TYPE, SIZE>(3, n_neurons_for_layer);

    std::vector<Value_Vec<TYPE>> xs = {
        {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};

    // desired target
    Value_Vec<TYPE> ys = {1.0, -1.0, -1.0, 1.0};

    std::cout << model; // to output the network shape

    std::cout << "The network hash: " << model.parameters().size()
              << " parameters\n\n";

    std::vector<std::vector<Value_Vec<TYPE>>> ypred;
    const double step_size = 0.001;

    std::cout << "Starting Training\n";
    std::cout << "----------------------------\n\n";

    for(size_t x = 0; x < 1000 ; x++){
        // Reset in case it is not the first loop
        auto loss = Value<TYPE>(0, "loss");
        ypred.clear();

        for (size_t i = 0; i < PASS_NUM; i++) {
            ypred.emplace_back(model(xs[i]));
            // Mean Squared Error
            loss += ((ypred[i][SIZE][0] - ys[i]) ^ 2);
            loss.backward();
            // The gradient is in the direction of increased loss
            for (auto &p : model.parameters()) {
                // Thus we have to decrease the value
                p->data += -(step_size * p->grad);
            }
            model.zero_grad();
        }
        std::cout << "The loss at step: " << x << " is: "<< loss.data << '\n';
    }

    // loss.draw_graph();
}
