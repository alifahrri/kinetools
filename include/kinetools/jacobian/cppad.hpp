#ifndef KINETOOLS_JACOBIAN_CPPAD_HPP
#define KINETOOLS_JACOBIAN_CPPAD_HPP

#include <cppad/cppad.hpp>
#include "kinetools/kinetools.hpp"

namespace kinetools {
    namespace jacobian {
        namespace cppad {

            template <typename F, typename IterableJoints>
            auto record_jacobian(const F &tf, IterableJoints &joints)
            {
                // static_assert(
                //     std::conjunction_v<
                //         utility::traits::is_callable<const F&, IterableJoints&>,
                //         utility::traits::is_indexable<IterableJoints&>
                //     >, "expects tf(joints) is valid, and joints are indexable, e.g. joints[0] is valid!"
                // );
                using scalar_t = std::decay_t<decltype(joints[0])>;
                // static_assert(
                //     std::conjunction_v<
                //         utility::traits::has_value_type<scalar_t>
                //     >, "expects joints[0] has value_type"
                // );
                // static_assert(
                //     std::conjunction_v<
                //         std::is_same<scalar_t,CppAD::AD<typename scalar_t::value_type>>
                //     >, "expects joints[0] has type of CppAD::AD<>"
                // );
                using base_t = typename scalar_t::value_type;
                CppAD::Independent(joints);
                auto p = tf(joints);
                // using base_t = typename std::decay_t<decltype(p)>::value_type;
                return CppAD::ADFun<base_t>(joints, p);
            }

            template <typename Vec, typename IterableTransforms, typename IterableJoints>
            auto translation_jacobian(IterableTransforms &tfs, IterableJoints &joints)
            {
                auto f = [=](auto joints) {
                    assert(std::size(joints)==std::size(tfs));
                    auto n = std::size(tfs) - 1;
                    auto px = utility::block(tfs[n](joints[n]), 0, 3, 4, 1);
                    auto base_tf = kinetools::base_transform(tfs,joints,n);
                    return static_cast<Vec>(base_tf*px);
                };
                auto F = record_jacobian(f, joints);
                return F;
            }

            template <typename Vec, typename IterableTransforms, typename IterableJointTypes>
            auto rotation_jacobian(IterableTransforms &tfs, const IterableJointTypes &joint_types)
            {
                assert(std::size(joint_types)==std::size(tfs));
                auto f = [=](auto &joints, size_t i, bool is_rot) -> Vec {
                    auto base_tf = kinetools::base_transform(tfs,joints,i);
                    auto eps = (is_rot ? 1. : 0.);
                    auto Rz = utility::block(base_tf, 0, 2, 3, 1) * eps;
                    return Rz;
                };
                return [=](auto &joints){
                    assert(std::size(joints)==std::size(tfs));
                    // auto M = Mat::Zero(3,std::size(tfs));
                    std::vector<Vec> Rzs;
                    for (size_t i=0; i<std::size(tfs); i++) {
                        auto Rz = f(joints, i, bool{joint_types[i]});
                        Rzs.push_back(Rz);
                    }
                    return Rzs;
                };
            }

        } // namespace cppad
    } // namespace jacobian
} // namespace kinetools

#endif // KINETOOLS_JACOBIAN_CPPAD_HPP