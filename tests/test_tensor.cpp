#include "autograd/tensor.hpp"
#include "autograd/ops.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

int main()
{
    using namespace ag;

    {
        std::vector<double> va(24), vb(24);
        for (size_t i = 0; i < 24; ++i)
        {
            va[i] = i;
            vb[i] = 2.0 * i;
        }
        auto a = std::make_shared<Tensor>(va, std::vector<size_t>{2, 3, 4});
        auto b = std::make_shared<Tensor>(vb, std::vector<size_t>{2, 3, 4});
        auto c = a + b;

        assert(c->shape().size() == 3);
        assert(c->shape()[0] == 2);
        assert(c->shape()[1] == 3);
        assert(c->shape()[2] == 4);

        for (size_t i = 0; i < 24; ++i)
            assert(c->values()[i] == va[i] + vb[i]);

        c->backward();
        for (size_t i = 0; i < 24; ++i)
        {
            assert(a->grads()[i] == 1.0);
            assert(b->grads()[i] == 1.0);
        }

        std::cout << "Test 1 (3D addition) passed\n";
    }

    {
        std::vector<double> va(48), vb(48);
        for (size_t i = 0; i < 48; ++i)
        {
            va[i] = 1.0 + i * 0.1;
            vb[i] = 0.5 + i * 0.05;
        }
        auto a = std::make_shared<Tensor>(va, std::vector<size_t>{2, 3, 4, 2});
        auto b = std::make_shared<Tensor>(vb, std::vector<size_t>{2, 3, 4, 2});
        auto c = a * b;

        assert(c->shape().size() == 4);

        for (size_t i = 0; i < 48; ++i)
            assert(std::abs(c->values()[i] - va[i] * vb[i]) < 1e-12);

        c->backward();
        for (size_t i = 0; i < 48; ++i)
        {
            assert(std::abs(a->grads()[i] - vb[i]) < 1e-12);
            assert(std::abs(b->grads()[i] - va[i]) < 1e-12);
        }

        std::cout << "Test 2 (4D multiplication) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0}, std::vector<size_t>{2});
        auto b = std::make_shared<Tensor>(std::vector<double>{4.0, 5.0}, std::vector<size_t>{2});

        auto ab = a * b; 
        auto ab_a = ab + a; 
        auto c = ab_a * b;

        assert(c->values()[0] == 40.0);
        assert(c->values()[1] == 90.0);

        c->backward();
 
        assert(a->grads()[0] == 20.0);
        assert(a->grads()[1] == 30.0);
        assert(b->grads()[0] == 18.0);
        assert(b->grads()[1] == 33.0);

        std::cout << "Test 3 (long chain) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(std::vector<double>{1.5, 2.5, 3.5}, std::vector<size_t>{3});
        auto b = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0, 4.0}, std::vector<size_t>{3});

        auto ab = a * b;
        auto c = ab + ab + a * a;

        c->backward();
        assert(a->grads()[0] == 7.0);
        assert(a->grads()[1] == 11.0);
        assert(a->grads()[2] == 15.0);
        assert(b->grads()[0] == 3.0);
        assert(b->grads()[1] == 5.0);
        assert(b->grads()[2] == 7.0);

        std::cout << "Test 4 (three shared uses) passed\n";
    }

    {
        const size_t N = 1000;
        std::vector<double> va(N), vb(N);
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < N; ++i)
        {
            va[i] = dist(gen);
            vb[i] = dist(gen);
        }

        auto a = std::make_shared<Tensor>(va, std::vector<size_t>{N});
        auto b = std::make_shared<Tensor>(vb, std::vector<size_t>{N});
        auto c = a * b + a + b;

        c->backward();

        for (size_t i = 0; i < N; ++i)
        {
            assert(std::abs(a->grads()[i] - (vb[i] + 1.0)) < 1e-12);
            assert(std::abs(b->grads()[i] - (va[i] + 1.0)) < 1e-12);
        }

        std::cout << "Test 5 (1000-element tensor) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0}, std::vector<size_t>{2});
        auto b = std::make_shared<Tensor>(std::vector<double>{3.0, 4.0}, std::vector<size_t>{2});

        for (int step = 0; step < 5; ++step)
        {
            auto c = a * b;
            c->backward();
            assert(a->grads()[0] == 3.0);
            assert(a->grads()[1] == 4.0);
            assert(b->grads()[0] == 1.0);
            assert(b->grads()[1] == 2.0);

            a->zero_grad();
            b->zero_grad();
            assert(a->grads()[0] == 0.0);
            assert(b->grads()[0] == 0.0);
        }

        std::cout << "Test 6 (training loop) passed\n";
    }

    {
        std::vector<double> va(24);
        for (size_t i = 0; i < 24; ++i)
            va[i] = i + 1.0;
        auto a = std::make_shared<Tensor>(va, std::vector<size_t>{2, 3, 4});
        auto c = a * 2.5 + 1.0;

        assert(c->shape().size() == 3);
        for (size_t i = 0; i < 24; ++i)
            assert(c->values()[i] == va[i] * 2.5 + 1.0);

        c->backward();
        for (size_t i = 0; i < 24; ++i)
            assert(a->grads()[i] == 2.5);

        std::cout << "Test 7 (3D scalar promotion) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0}, std::vector<size_t>{2});
        auto b = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0, 3.0}, std::vector<size_t>{3});

        bool caught_add = false;
        try
        {
            auto c = a + b;
        }
        catch (const std::runtime_error &)
        {
            caught_add = true;
        }
        assert(caught_add);

        bool caught_mul = false;
        try
        {
            auto c = a * b;
        }
        catch (const std::runtime_error &)
        {
            caught_mul = true;
        }
        assert(caught_mul);

        std::cout << "Test 8 (shape mismatch throws) passed\n";
    }

    {
        std::vector<double> va(30), vb(30);
        for (size_t i = 0; i < 30; ++i)
        {
            va[i] = i * 0.1;
            vb[i] = 1.0 - i * 0.05;
        }
        auto a = std::make_shared<Tensor>(va, std::vector<size_t>{2, 5, 3});
        auto b = std::make_shared<Tensor>(vb, std::vector<size_t>{2, 5, 3});

        auto c = a * b;
        c->backward();

        for (size_t i = 0; i < 30; ++i)
        {
            assert(std::abs(a->grads()[i] - vb[i]) < 1e-12);
            assert(std::abs(b->grads()[i] - va[i]) < 1e-12);
        }

        std::cout << "Test 9 (3D non-square) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(std::vector<double>{1.0, 2.0}, std::vector<size_t>{2});
        auto c = a + 2.0 + 3.0 + 4.0;

        assert(c->values()[0] == 10.0);
        assert(c->values()[1] == 11.0);

        c->backward();
        assert(a->grads()[0] == 1.0);
        assert(a->grads()[1] == 1.0);

        std::cout << "Test 10 (chained scalar additions) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(std::vector<double>{2.0, 3.0}, std::vector<size_t>{2});

        auto c1 = 4.0 * a; 
        assert(c1->values()[0] == 8.0);
        assert(c1->values()[1] == 12.0);
        c1->backward();
        assert(a->grads()[0] == 4.0);
        assert(a->grads()[1] == 4.0);

        a->zero_grad();

        auto c2 = 5.0 + a; 
        assert(c2->values()[0] == 7.0);
        assert(c2->values()[1] == 8.0);
        c2->backward();
        assert(a->grads()[0] == 1.0);
        assert(a->grads()[1] == 1.0);

        std::cout << "Test 11 (scalar on left) passed\n";
    }

    {
        auto x = std::make_shared<Tensor>(
            std::vector<double>{1.0, 2.0, 3.0, 4.0},
            std::vector<size_t>{4});

        auto y = sum(x);
        assert(y->shape().size() == 1);
        assert(y->shape()[0] == 1);
        assert(y->values()[0] == 10.0);

        y->backward();
        for (size_t i = 0; i < 4; ++i)
            assert(x->grads()[i] == 1.0);

        std::cout << "Test 12 (sum 1D) passed\n";
    }

    {
        auto x = std::make_shared<Tensor>(
            std::vector<double>{2.0, 4.0, 6.0, 8.0},
            std::vector<size_t>{4});

        auto y = mean(x);
        assert(y->values()[0] == 5.0);

        y->backward();
        for (size_t i = 0; i < 4; ++i)
            assert(std::abs(x->grads()[i] - 0.25) < 1e-12);

        std::cout << "Test 13 (mean 1D) passed\n";
    }

    {
        auto x = std::make_shared<Tensor>(
            std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            std::vector<size_t>{2, 3});

        auto y = sum(x);
        assert(y->values()[0] == 21.0);

        y->backward();
        for (size_t i = 0; i < 6; ++i)
            assert(x->grads()[i] == 1.0);

        std::cout << "Test 14 (sum 2D) passed\n";
    }

    {
        auto x = std::make_shared<Tensor>(
            std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
            std::vector<size_t>{2, 3});

        auto y = mean(x);
        assert(std::abs(y->values()[0] - 3.5) < 1e-12);  

        y->backward();
        for (size_t i = 0; i < 6; ++i)
            assert(std::abs(x->grads()[i] - 1.0 / 6.0) < 1e-12);

        std::cout << "Test 15 (mean 2D) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(
            std::vector<double>{1.0, 2.0, 3.0},
            std::vector<size_t>{3});
        auto b = std::make_shared<Tensor>(
            std::vector<double>{4.0, 5.0, 6.0},
            std::vector<size_t>{3});

        auto c = a * b;              
        auto y = sum(c);            

        assert(y->values()[0] == 32.0);

        y->backward();
        assert(a->grads()[0] == 4.0);
        assert(a->grads()[1] == 5.0);
        assert(a->grads()[2] == 6.0);
        assert(b->grads()[0] == 1.0);
        assert(b->grads()[1] == 2.0);
        assert(b->grads()[2] == 3.0);

        std::cout << "Test 16 (sum of a*b) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(
            std::vector<double>{1.0, 2.0, 3.0, 4.0},
            std::vector<size_t>{4});
        auto b = std::make_shared<Tensor>(
            std::vector<double>{5.0, 6.0, 7.0, 8.0},
            std::vector<size_t>{4});

        auto c = a * b;               
        auto y = mean(c);            

        assert(std::abs(y->values()[0] - 17.5) < 1e-12);

        y->backward();
        for (size_t i = 0; i < 4; ++i) {
            assert(std::abs(a->grads()[i] - b->values()[i] / 4.0) < 1e-12);
            assert(std::abs(b->grads()[i] - a->values()[i] / 4.0) < 1e-12);
        }

        std::cout << "Test 17 (mean of a*b) passed\n";
    }

    {
        auto a = std::make_shared<Tensor>(
            std::vector<double>{2.0, 3.0, 4.0},
            std::vector<size_t>{3});

        auto c = a * a;              
        auto y = sum(c);             

        assert(y->values()[0] == 29.0);

        y->backward();
        assert(a->grads()[0] == 4.0);
        assert(a->grads()[1] == 6.0);
        assert(a->grads()[2] == 8.0);

        std::cout << "Test 18 (sum of a*a) passed\n";
    }

    {
        auto pred = std::make_shared<Tensor>(
            std::vector<double>{1.0, 2.0, 3.0},
            std::vector<size_t>{3});
        auto target = std::make_shared<Tensor>(
            std::vector<double>{1.0, 2.0, 5.0},
            std::vector<size_t>{3});

        auto neg_target = target * std::make_shared<Tensor>(
            std::vector<double>{-1.0, -1.0, -1.0},
            std::vector<size_t>{3});
        auto diff = pred + neg_target;       

        auto sq = diff * diff;               
        auto loss = mean(sq);                

        assert(std::abs(loss->values()[0] - 4.0 / 3.0) < 1e-12);

        loss->backward();
        assert(std::abs(pred->grads()[0] - 0.0) < 1e-12);
        assert(std::abs(pred->grads()[1] - 0.0) < 1e-12);
        assert(std::abs(pred->grads()[2] - (-4.0 / 3.0)) < 1e-12);

        std::cout << "Test 19 (MSE loss) passed\n";
    }

    {
        const size_t N = 120;  
        std::vector<double> vals(N);
        double expected = 0.0;
        for (size_t i = 0; i < N; ++i) {
            vals[i] = 0.5 * (i + 1);
            expected += vals[i];
        }

        auto x = std::make_shared<Tensor>(vals, std::vector<size_t>{2, 3, 4, 5});
        auto y = sum(x);

        assert(std::abs(y->values()[0] - expected) < 1e-9);

        y->backward();
        for (size_t i = 0; i < N; ++i)
            assert(x->grads()[i] == 1.0);

        std::cout << "Test 20 (sum on 4D) passed\n";
    }


    std::cout << "\nAll harder tensor tests passed!\n";
    return 0;
}
