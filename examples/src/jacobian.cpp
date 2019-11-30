#include <cppad/cppad.hpp>
#include <vector>

namespace {
    auto jacobian() {
        using CppAD::AD;
        using CppAD::exp;
        using CppAD::sin;
        using CppAD::cos;

        size_t n{2};
        CppAD::vector<AD<double>> X(n);
        X[0] = 1.;
        X[1] = 2.;

        CppAD::Independent(X);
        AD<double> square = X[0] * X[0];

        size_t m{3};
        CppAD::vector<AD<double>> Y(m);
        Y[0] = square * exp(X[1]);
        Y[1] = square * sin(X[1]);
        Y[2] = square * cos(X[1]);

        CppAD::ADFun<double> f(X, Y);

        std::vector<double> x(n);
        x[0] = 2.;
        x[1] = 1.;

        std::vector<double> j(m*n);
        j = f.Jacobian(x);

        std::vector<double> e(m*n);
        /* partial derivative for x[0] */
        e[0] = 2. * x[0] * exp(x[1]);
        e[1] = 2. * x[0] * sin(x[1]);
        e[2] = 2. * x[0] * cos(x[1]);

        e[3] = x[0]*x[0] * exp(x[1]);
        e[4] = x[0]*x[0] * cos(x[1]);
        e[5] = -x[0]*x[0] * sin(x[1]);

        for (size_t i=0; i<j.size(); i++)
            std::cout << j[i] << ", " << e[i] << std::endl;
    }
}

int main(int argc, char**argv)
{
    ::jacobian();
    return 0;
}