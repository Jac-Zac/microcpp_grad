#include <micrograd/engine.hpp>

Value<double> compute(Value<double>& d,const Value<double>& tmp3){
    Value<double>* copy = new Value<double>(d.data, "copy of forward", d.m_op , d.m_prev);
    Value<double>* tmp3_copy = new Value<double>(tmp3.data, "copy of tmp", tmp3.m_op , tmp3.m_prev);

    return (*copy + *tmp3_copy);
}

int main() {
    std::vector<Value<double>> a = {
        Value<double>(5.0, "a_1"),
        Value<double>(10.0, "a_2"),
    };

    std::vector<Value<double>> b = {
        Value<double>(-2.0, "b_1"),
        Value<double>(-4.0, "b_2"),
    };

    std::vector<Value<double>> c;
    std::vector<Value<double>> y;

    for(int i = 0 ; i < 2; i++){
        c.emplace_back((a[i] * b[i]));
    }

    for(int i = 0 ; i < 2; i++){
        y.emplace_back(c[i]^2);
    }

    auto forward = Value<double>(0, "forward");

    auto forward_copy = forward;

    forward = forward_copy +y[0];

    auto new_forward_copy = forward;

    forward = new_forward_copy + y[1];

    /* for(size_t i = 0 ; i < 2;  i++){ */
        /* forward += (c[i] * d); */
    /* } */

    forward.backward();
    forward.draw_graph();
}
