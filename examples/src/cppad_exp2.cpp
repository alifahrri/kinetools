#include <cppad/cppad.hpp>
#include <iostream>

namespace fun {

    template <typename Type>
    auto exp2(const Type &x)
    {
        Type v1 = x;
        Type v2 = Type{1} + v1;
        Type v3 = v1 * v1;
        Type v4 = v3 / Type{2};
        Type v5 = v2 + v4;
        return v5;
    }

} // namespace fun

int main(int argc, char **argv)
{
    using CppAD::AD;
    using CppAD::vector;

    size_t n{1};
    vector<AD<double>> X(n);
    X[0] = .5;

    CppAD::Independent(X);

    AD<double> x = X[0];
    AD<double> apx = fun::exp2(x);

    size_t m{1};
    vector<AD<double>> Y(m);
    Y[0] = apx;

    CppAD::ADFun<double> f(X, Y);

    // first order forward sweep that computes
    // partial of exp2(x) with respect to x
    vector<double> dx(n);
    vector<double> dy(m);
    dx[0] = 1.;
    dy = f.Forward(1, dx);
    std::cout << "first order forward : " << dy << std::endl;

    // first order reverse sweep that computes the derivatives
    vector<double>  w(m);
    vector<double> dw(n);
    w[0] = 1.;
    dw = f.Reverse(1, w);
    std::cout << "first order reverse : " << dw << std::endl;

    // second order forward sweep
    vector<double> x2(n);
    vector<double> y2(m);
    x2[0] = 0.;
    y2 = f.Forward(2, x2);
    std::cout << "second order forward : " << y2 << std::endl;

    dw.resize(2*n);
    dw = f.Reverse(2,w);
    std::cout << "second order forward : " << dw << std::endl;
}