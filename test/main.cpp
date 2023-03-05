#include <micrograd/engine.hpp>

Value<double> compute(Value<double>& d,const Value<double>& tmp3){
    Value<double>* copy = new Value<double>(d.data, "copy of forward", d.m_op , d.m_prev);
    Value<double>* tmp3_copy = new Value<double>(tmp3.data, "copy of tmp", tmp3.m_op , tmp3.m_prev);

    return (*copy + *tmp3_copy);
}

std::vector<Value<double>*> start(){
        // Testing
    std::vector<Value<double>*> a = {
        new Value<double>(-1.0, "a_1"),
        new Value<double>(+2.0, "a_2"),
        new Value<double>(-3.0, "a_3"),
        new Value<double>(+4.0, "a_4"),
    };

    std::vector<Value<double>*> b = {
        new Value<double>(-1.0, "b_1"),
        new Value<double>(+2.0, "b_2"),
        new Value<double>(-3.0, "b_3"),
        new Value<double>(+4.0, "b_4"),
    };

    std::vector<Value<double>*> c;

    for(int i = 0 ; i < 4; i++){
        Value<double>* tmp = new Value<double>(-1.0, "tmp");
        *tmp = (*a[i] * *b[i]) + 5;
        tmp->label = "c";
        c.emplace_back(tmp);
    }
    return c;
}

int main() {

    auto d = Value<double>(5, "d");
    auto bias = Value<double>(10, "bias");

    auto forward = Value<double>(0, "forward");

    std::vector<Value<double>*> c = std::move(start());

    for(size_t i = 0 ; i < 4;  i++){
        forward = compute(forward,(*c[i] * d));
        /* forward += *c[i] * d; */
    }

    forward += bias;

    forward.backward();
    forward.draw_graph();
}
