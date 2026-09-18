#include "autograd/value.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

int main() {
    using namespace ag;

    std::cout << "Running Backward Tests...\n" << std::endl;

    {
        auto a = std::make_shared<Value>(2.0);
        auto b = std::make_shared<Value>(3.0);
        auto c = a + b;

        c->zero_grad(); 
        c->backward();

        assert(a->grad() == 1.0);
        assert(b->grad() == 1.0);
        std::cout << "Test 1 (Addition) PASSED" << std::endl;
    }

    {
        auto a = std::make_shared<Value>(2.0);
        auto b = std::make_shared<Value>(3.0);
        auto c = a * b;

        c->zero_grad();
        c->backward();

        assert(a->grad() == 3.0);
        assert(b->grad() == 2.0);
        std::cout << "Test 2 (Multiplication) PASSED" << std::endl;
    }
    {
        auto a = std::make_shared<Value>(2.0);
        auto c = a * a; 

        c->zero_grad();
        c->backward();

        assert(a->grad() == 4.0);
        std::cout << "Test 3 (Repeated Variable) PASSED" << std::endl;
    }
    {
        auto a = std::make_shared<Value>(2.0);
        auto b = std::make_shared<Value>(3.0);
        auto mult = a * b;
        auto c = mult + a;

        c->zero_grad();
        c->backward();

        assert(a->grad() == 4.0);
        assert(b->grad() == 2.0);
        std::cout << "Test 4 (Mixed Operations) PASSED" << std::endl;
    }
    {
        auto a = std::make_shared<Value>(2.0);
        auto b = std::make_shared<Value>(3.0);
        auto sum = a + b;
        auto c = sum * a;

        c->zero_grad();
        c->backward();
        assert(std::abs(a->grad() - 7.0) < 1e-9);
        assert(std::abs(b->grad() - 2.0) < 1e-9);
        std::cout << "Test 5 (Branching) PASSED" << std::endl;
    }
    {
        auto a = std::make_shared<Value>(2.0);
        auto c = a * 2.0;

        c->zero_grad();
        c->backward();

        assert(a->grad() == 2.0);
        std::cout << "Test 6 (Constant Promotion) PASSED" << std::endl;
    }
    {
        auto a = std::make_shared<Value>(2.0);
        auto b = std::make_shared<Value>(3.0);
        auto c = a * b;

        c->zero_grad();
        c->backward();
        assert(a->grad() == 3.0);
        assert(b->grad() == 2.0);

        c->zero_grad();
        assert(a->grad() == 0.0);
        assert(b->grad() == 0.0);

        c->backward();
        assert(a->grad() == 3.0);
        assert(b->grad() == 2.0);

        std::cout << "Test 7 (Zero Grad Reset) PASSED" << std::endl;
    }
    {
        auto a = std::make_shared<Value>(2.0);
        auto c = a * a;
        c->backward();  

        bool exception_thrown = false;
        try {
            c->backward();
        } catch (const std::runtime_error& e) {
            exception_thrown = true;
        }

        assert(exception_thrown);
        std::cout << "Test 8 (Exception Handling) PASSED" << std::endl;
    }

    std::cout << "\nAll backward tests passed successfully!" << std::endl;
    return 0;
}