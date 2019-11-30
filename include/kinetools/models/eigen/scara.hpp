#ifndef KINETOOLS_SCARA_HPP
#define KINETOOLS_SCARA_HPP

#include <array>
#include <tuple>
#include <cmath>

#include <Eigen/Dense>

#include <cppad/example/cppad_eigen.hpp>
#include "kinetools/kinetools.hpp"
#include "kinetools/jacobian.hpp"
#include "kinetools/joints/eigen.hpp"

namespace kinetools {
    
    namespace ad = jacobian::cppad;

    template <typename T>
    struct Scara {
        using ad_scalar_t = CppAD::AD<T>;
        using ad_joint_t = joints::Vec<ad_scalar_t>;
        using transforms_t = std::array<joints::DHJoint<ad_scalar_t>,4>;
        using trans_jacobian_t = CppAD::ADFun<T>;
        using rot_jacobian_t = decltype(
            ad::rotation_jacobian<ad_joint_t>(
                std::declval<transforms_t&>(), std::declval<std::array<bool,4>>()
        ));

        Scara(const std::array<std::array<T,3>,4> &config);

        auto forward(const std::array<T,4> &joints);
        template <typename Joints>
        auto forward(const Joints &joints) {
            auto base_tfs = kinetools::base_transforms(this->transforms,joints);
            return base_tfs;        
        }
        auto jacobian(const std::array<T,4> &joints)
        {
            CppAD::vector<T> x(std::size(joints));
            for (size_t i=0; i<std::size(joints); i++)
                x[i] = joints[i];
            auto v = Jv.Jacobian(x);
            auto w = Jw(x);
            joints::Mat<T> J(6,4);
            J << v[0], v[1], v[2], v[3],
                v[4], v[5], v[6],  v[7],
                v[8], v[9], v[10], v[11],
                CppAD::Value(w[0][0]), CppAD::Value(w[1][0]), CppAD::Value(w[2][0]), CppAD::Value(w[3][0]),
                CppAD::Value(w[0][1]), CppAD::Value(w[1][1]), CppAD::Value(w[2][1]), CppAD::Value(w[3][1]),
                CppAD::Value(w[0][2]), CppAD::Value(w[1][2]), CppAD::Value(w[2][2]), CppAD::Value(w[3][2]);
            return J;
        }

        auto inverse(const joints::Vec<T> &p_ref, const joints::Mat<T> &R_ref, const joints::Vec<T> &q_init)
        {
            auto fwd = [&](const joints::Vec<T> &q){
                assert(q.rows()==4);
                assert(q.cols()==1);
                std::array<double,4> q_array{
                    q[0], q[1], q[2], q[3]
                };
                auto transforms = this->forward(q_array);
                auto Tb = transforms.back();
                auto M = joints::Mat<T>(4,4);
                for (size_t i=0; i<4; i++)
                    for (size_t j=0; j<4; j++)
                        M(i,j) = CppAD::Value(Tb(i,j));
                return M;
            };
            auto jac = [&](const joints::Vec<T> &q){
                assert(q.rows()==4);
                assert(q.cols()==1);
                std::array<double,4> q_array{
                    q[0], q[1], q[2], q[3]
                };
                return this->jacobian(q_array);
            };
            auto q = inverse_kinematic_lm(p_ref,R_ref,fwd,jac,q_init);
            return q;
        }

        transforms_t transforms;
        trans_jacobian_t Jv;
        rot_jacobian_t Jw;
        
    }; // class Scara
} // namespace kinetools

template <typename T>
kinetools::Scara<T>::Scara(const std::array<std::array<T,3>,4> &config)
    : transforms{
        joints::DHJoint<ad_scalar_t>({config[0][0],config[0][1],config[0][2]},joints::JointType::Rotational),
        joints::DHJoint<ad_scalar_t>({config[1][0],config[1][1],config[1][2]},joints::JointType::Rotational),
        joints::DHJoint<ad_scalar_t>({config[2][0],config[2][1],config[2][2]},joints::JointType::Prismatic),
        joints::DHJoint<ad_scalar_t>({config[3][0],config[3][1],config[3][2]},joints::JointType::Rotational)
    },
    Jw(ad::rotation_jacobian<ad_joint_t>(
        transforms, std::array<bool,4>{true, true, false, true})
    )
{
    ad_joint_t joints(std::size(transforms));
    /* arbitrary number for recording */
    joints[0] = M_PI/2;
    joints[1] = M_PI/2;
    joints[2] = 0.5/3;
    joints[3] = M_PI/2;
    Jv = ad::translation_jacobian<ad_joint_t>(transforms, joints);
}

template <typename T>
auto kinetools::Scara<T>::forward(const std::array<T,4> &joints)
{
    auto base_tfs = kinetools::base_transforms(this->transforms,joints);
    return base_tfs;
}

#endif // KINEMATICS_SCARA_HPP