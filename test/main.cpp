#include <micrograd/engine.hpp>

Value<double> compute(Value<double>& d, Value<double>& tmp3){
    Value<double>* copy = new Value<double>(d.data, "copy of d", d.m_op , d.m_prev);
    d = *copy + tmp3;
    return d;
}

int main() {
    // Testing
    auto a = Value<double>(-4.0, "a");
    auto b = Value<double>(2.0, "b");
    /* auto c = a * 2; */
    /* c.label = "c"; */
    auto d = Value<double>(0);
    d.label = "d";

    auto tmp1 = a;
    tmp1.label = "tmp 1 = a";
    auto tmp2 = b;
    tmp2.label = "tmp 2 = b";

    /* auto tmp3 = c; */
    /* tmp2.label = "tmp 3 = c"; */
    {

        auto tmp3  = (tmp2 * tmp1)^2;
        tmp3.label = "tmp3 computed";
        d = compute(d, tmp3);
    }

    /* d += (b - a) ^ 2; */

    /* d += (c - a) ^ 2; */
    d.backward();
    d.draw_graph();
}
