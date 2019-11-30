#include <Eigen/Dense>
#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>
#include "kinetools/kinetools.hpp"

using CppAD::AD;
using Eigen::Matrix;
using Eigen::Dynamic;

using Mat = Matrix<AD<double>, Dynamic, Dynamic>;
template <typename T>
using Vec = Matrix<AD<T>, Dynamic, 1>;

template <typename T>
struct Translation {
    Translation(const T &dn) : dn(dn) {};
    auto operator()(const Vec<T> &v) {
        auto M = kinetools::dh::transz<Mat>(dn);
        return kinetools::utility::mul(M,v);
    }
    const T dn;
};

template <typename T>
struct Rotation {
    Rotation(const T &theta) : theta(theta) {};
    auto operator()(const Vec<T> &v) {
        auto M = kinetools::dh::rotz<Mat>(theta);
        return kinetools::utility::mul(M,v);
    }
    const T theta;
};

template <typename T, typename R>
struct Transform {
    Transform(const T &d, const R &theta, const T &r, const R &alpha) 
        : d(d), theta(theta), r(r), alpha(alpha) {}
    auto operator()(const Vec<T> &v) {
        Mat M = kinetools::dh::transform<Mat>(d,theta,r,alpha);
        return kinetools::utility::mul(M,v);
    }
    const T d;
    const R theta;
    const T r;
    const R alpha;
};

int main(int argc, char**argv)
{
    double dn{12.0};
    auto transz = kinetools::dh::transz<Mat>(dn);
    for (size_t i=0; i<4; i++)
        for (size_t j=0; j<4; j++)
            std::cout << transz(i,j) << std::endl;
    
    {
        std::cout << "translation" << std::endl;
        size_t n{4}; size_t m{n};
        Vec<double> X(n);
        Vec<double> Y(m);
        X[0] = 0.5;
        X[1] = 1.5;
        X[2] = 2.5;
        X[3] = 1.;

        auto trans = Translation<double>(3.);

        CppAD::Independent(X);

        Y = trans(X);
        
        CppAD::ADFun<double> f(X, Y);

        for (size_t i=0; i<n; i++)
            std::cout << X[i] << std::endl;
        for (size_t i=0; i<n; i++)
            std::cout << Y[i] << std::endl;

        CppAD::vector<double> x(n);
        x[0] = 0.5;
        x[1] = 0.5;
        x[2] = 0.5;
        x[3] = 1;
        auto J = f.Jacobian(x);
        auto df = f.Forward(1,x);
        auto dw = f.Reverse(1,x);
        std::cout << J << std::endl;
        std::cout << df << std::endl;
        std::cout << dw << std::endl;
    }

    {
        std::cout << "compute transform" << std::endl;
        size_t n{4}; size_t m{n};
        Vec<double> X(n);
        Vec<double> Y(m);
        X[0] = 0.5;
        X[1] = 1.5;
        X[2] = 2.5;
        X[3] = 1.;

        auto trans = Transform<double,double>(3., 0.5, 3., 0.5);

        CppAD::Independent(X);

        Y = trans(X);
        
        CppAD::ADFun<double> f(X, Y);

        for (size_t i=0; i<n; i++)
            std::cout << X[i] << std::endl;
        for (size_t i=0; i<n; i++)
            std::cout << Y[i] << std::endl;

        CppAD::vector<double> x(n);
        x[0] = 0.5;
        x[1] = 0.5;
        x[2] = 0.5;
        x[3] = 1;
        auto J = f.Jacobian(x);
        auto df = f.Forward(1,x);
        auto dw = f.Reverse(1,x);
        std::cout << J << std::endl;
        std::cout << df << std::endl;
        std::cout << dw << std::endl;
    }
    return 0;
}