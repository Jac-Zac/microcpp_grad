#include <micrograd/engine.hpp>

Value<double> compute(Value<double>& d,const Value<double>& tmp3){
    Value<double>* copy = new Value<double>(d.data, "copy of forward", d.m_op , d.m_prev);
    Value<double>* tmp3_copy = new Value<double>(tmp3.data, "copy of tmp", tmp3.m_op , tmp3.m_prev);

    return (*copy + *tmp3_copy);
}

int main() {
    // Testing
    std::vector<Value<double>> a = {
        Value<double>(-1.0, "a_1"),
        Value<double>(+2.0, "a_2"),
        Value<double>(-3.0, "a_3"),
        Value<double>(+4.0, "a_4")
    };

    auto d = Value<double>(5, "d");
    auto bias = Value<double>(10, "bias");

    auto forward = Value<double>(0, "forward");

    for(size_t i = 0 ; i < 4; i++){
        forward = compute(forward,(a[i] * d));
    }

    forward += bias;

    forward.backward();
    forward.draw_graph();
}
