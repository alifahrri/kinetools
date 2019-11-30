#include <Eigen/Dense>
#include <cppad/example/cppad_eigen.hpp>

#include "kinetools/jacobian.hpp"
#include "kinetools/kinetools.hpp"
#include "kinetools/joints/eigen.hpp"

#include <array>
#include <vector>
#include <cmath>
#include <iostream>

int main(int argc, char**argv)
{
    namespace kt = kinetools;
    namespace jc = kt::jacobian;
    namespace ad = jc::cppad;
    using CppAD::AD;
    using rJoint = kt::joints::Rotational<AD<double>>;
    using pJoint = kt::joints::Prismatic<AD<double>>;
    using Joint = kt::joints::Joint<AD<double>>;
    using DHJoint = kt::joints::DHJoint<AD<double>>;
    std::vector<DHJoint> tfs{
        DHJoint({1.0,1.0,0.0},kt::joints::JointType::Rotational),
        DHJoint({0.0,1.0,M_PI},kt::joints::JointType::Rotational),
        DHJoint({0.0,0.0,0.0},kt::joints::JointType::Prismatic),
        DHJoint({1.0,0.0,0.0},kt::joints::JointType::Rotational),
    };

    kt::joints::Vec<AD<double>> joints(std::size(tfs));
    joints[0] = M_PI/2;
    joints[1] = M_PI/2;
    joints[2] = 0.5/3;
    joints[3] = M_PI/2;

    auto f = [&](auto &joints) {
        assert(std::size(joints)==std::size(tfs));
        auto n = std::size(tfs) - 1;
        auto px = kt::utility::block(tfs[n](joints[n]), 0, 3, 4, 1);
        auto base_tf = kinetools::base_transform(tfs,joints,n);
        return static_cast<kt::joints::Vec<AD<double>>>(base_tf*px);
    };
    auto F = ad::record_jacobian(f, joints);
    std::vector<bool> joint_types {
        true, true, false, true
    };
    auto Jw = ad::rotation_jacobian<kt::joints::Vec<AD<double>>>(tfs, joint_types);
    {
        CppAD::vector<double> x(std::size(tfs));
        x[0] = M_PI/4;
        x[1] = M_PI/4;
        x[2] = 1.25;
        x[3] = M_PI/4;
        auto f = F.Forward(0,x);
        auto J = F.Jacobian(x);
        std::cout << f << std::endl;
        std::cout << J << std::endl;
        auto M = Jw(x);
        for (size_t i=0; i<M.size(); i++)
            std::cout << M[i] << std::endl;
    }
}