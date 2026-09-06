#include "autograd/value.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace ag;

    // 1. Create leaf nodes
    auto a = std::make_shared<Value>(2.0);
    auto b = std::make_shared<Value>(3.0);

    // 2. Forward operations
    auto sum = a + b;       // 2.0 + 3.0 = 5.0
    auto product = a * b;   // 2.0 * 3.0 = 6.0

    // 3. Check forward values
    assert(sum->value() == 5.0);
    assert(product->value() == 6.0);

    // 4. Check operation strings
    assert(sum->oper() == "+");
    assert(product->oper() == "*");
    assert(a->oper() == "leaf");
    assert(b->oper() == "leaf");

    // 5. Check the graph structure (prev)
    // sum's children are {a, b}
    const auto& sum_prev = sum->prev();
    assert(sum_prev.size() == 2);
    assert(sum_prev[0] == a);
    assert(sum_prev[1] == b);

    // product's children are {a, b}
    const auto& prod_prev = product->prev();
    assert(prod_prev.size() == 2);
    assert(prod_prev[0] == a);
    assert(prod_prev[1] == b);

    // 6. More complex: (a + b) * (a - b) ? Not implemented yet, but we can do (a + b) * a
    auto sum2 = a + b;      // 5.0
    auto product2 = sum2 * a; // 5.0 * 2.0 = 10.0
    assert(product2->value() == 10.0);
    assert(product2->oper() == "*");
    const auto& prod2_prev = product2->prev();
    assert(prod2_prev.size() == 2);
    assert(prod2_prev[0] == sum2);
    assert(prod2_prev[1] == a);

    // 7. Constant promotion
    auto with_double = a + 4.0;  // 2.0 + 4.0 = 6.0
    assert(with_double->value() == 6.0);
    assert(with_double->oper() == "+");
    const auto& with_double_prev = with_double->prev();
    assert(with_double_prev.size() == 2);
    assert(with_double_prev[0] == a);
    // The second child is a temporary Value(4.0) – we can't assert exact pointer,
    // but we can check its value is 4.0.
    assert(with_double_prev[1]->value() == 4.0);
    assert(with_double_prev[1]->oper() == "leaf");

    // 8. Mixed constants
    auto left_double = 2.5 * b; // 2.5 * 3.0 = 7.5
    assert(left_double->value() == 7.5);
    assert(left_double->oper() == "*");
    const auto& left_prev = left_double->prev();
    assert(left_prev.size() == 2);
    assert(left_prev[0]->value() == 2.5);
    assert(left_prev[1] == b);

    std::cout << "All forward-pass tests passed!" << std::endl;
    return 0;
}