#include "autograd/tensor.hpp"
#include "autograd/ops.hpp" 
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using ag::Tensor;
using Ptr = std::shared_ptr<Tensor>;


static Ptr make(const std::vector<double>& v, const std::vector<size_t>& s)
{
    return std::make_shared<Tensor>(v, s);
}

static bool nearly_equal(double a, double b, double eps = 1e-12)
{
    return std::fabs(a - b) < eps;
}

static bool values_eq(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!nearly_equal(a[i], b[i])) return false;
    return true;
}

static bool shape_eq(const std::vector<size_t>& a, const std::vector<size_t>& b)
{
    return a == b;
}


static void test_2d_forward()
{

    auto A = make({1, 2, 3, 4, 5, 6}, {2, 3});

    auto T = A->transpose(0, 1);

    assert(shape_eq(T->shape(), {3, 2}));
    assert(values_eq(T->values(), {1, 4, 2, 5, 3, 6}));

    auto T2 = A->transpose();
    assert(shape_eq(T2->shape(), {3, 2}));
    assert(values_eq(T2->values(), T->values()));
}


static void test_1d_noop()
{
    auto A = make({1, 2, 3}, {3});
    auto T = A->transpose();          

    assert(T.get() == A.get());        
    assert(shape_eq(T->shape(), {3}));
    assert(values_eq(T->values(), {1, 2, 3}));
}

static void test_3d_axes_0_2()
{
    std::vector<double> v(24);
    for (size_t i = 0; i < 24; ++i) v[i] = static_cast<double>(i);
    auto A = make(v, {2, 3, 4});

    auto T = A->transpose(0, 2);       

    assert(shape_eq(T->shape(), {4, 3, 2}));

    assert(nearly_equal(T->values()[23], 23.0));


    assert(nearly_equal(T->values()[14], 6.0));


    assert(nearly_equal(T->values()[1], 12.0));
}

static void test_3d_axes_1_2()
{
    std::vector<double> v(24);
    for (size_t i = 0; i < 24; ++i) v[i] = static_cast<double>(i);
    auto A = make(v, {2, 3, 4});
    auto T = A->transpose(1, 2);       
    assert(shape_eq(T->shape(), {2, 4, 3}));
    assert(nearly_equal(T->values()[23], 23.0));
}

static void test_involution_2d()
{
    auto A = make({1, 2, 3, 4, 5, 6}, {2, 3});
    auto T  = A->transpose(0, 1);
    auto TT = T->transpose(0, 1);

    assert(shape_eq(TT->shape(), A->shape()));
    assert(values_eq(TT->values(), A->values()));
}

static void test_backward_2d()
{

    auto A = make({1, 2, 3, 4, 5, 6}, {2, 3});

    auto T = A->transpose(0, 1);       
    auto s = ag::sum(T);               

    s->backward();

    assert(shape_eq(A->grads().size() == 6 ? std::vector<size_t>{2,3}
                                           : std::vector<size_t>{0}, {2,3}));
    for (double g : A->grads())
        assert(nearly_equal(g, 1.0));
}

static void test_backward_weighted()
{
    auto A = make({1, 2, 3, 4, 5, 6}, {2, 3});
    auto W = make({10, 20,
                   30, 40,
                   50, 60}, {3, 2});
    auto T  = A->transpose(0, 1);      
    auto y  = ag::sum(W * T);         

    y->backward();
    std::vector<double> expected = {10, 30, 50,
                                    20, 40, 60};
    assert(values_eq(A->grads(), expected));
}


static void test_bad_axes()
{
    auto A = make({1, 2, 3, 4}, {2, 2});

    bool threw = false;
    try { A->transpose(0, 0); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);

    threw = false;
    try { A->transpose(0, 5); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);

    threw = false;
    try { A->transpose(7, 0); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
}


int main()
{
    test_2d_forward();
    test_1d_noop();
    test_3d_axes_0_2();
    test_3d_axes_1_2();
    test_involution_2d();
    test_backward_2d();
    test_backward_weighted();
    test_bad_axes();

    std::cout << "all transpose tests passed\n";
    return 0;
}