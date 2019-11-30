#ifndef KINETOOLS_JOINTS_EIGEN_HPP
#define KINETOOLS_JOINTS_EIGEN_HPP

#include "kinetools/kinetools.hpp"
#include <variant>

namespace kinetools {
    namespace joints {

        template <typename T>
        using Mat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
        template <typename T>
        using Vec = Eigen::Matrix<T,Eigen::Dynamic,1>;

        template <typename T>
        struct Prismatic {
            Prismatic(const T d) : d(d) {}
            T d;
        };

        template <typename T>
        struct Rotational {
            Rotational(const T theta) : theta(theta) {}
            T theta;
        };

        template <typename T>
        using Joint = std::variant<std::monostate,Prismatic<T>,Rotational<T>>;

        template <typename T>
        struct JointVisitor {
            JointVisitor(const std::array<T,3> &constants) 
                : constants(constants) {}
            auto operator()(const Prismatic<T> &joint) const -> Mat<T> {
                using matrix_type = Mat<T>;
                const auto [theta,r,alpha] = constants;
                auto M = kinetools::dh::transform<matrix_type>(joint.d,theta,r,alpha);
                return M;
            }
            auto operator()(const Rotational<T> &joint) const -> Mat<T> {
                using matrix_type = Mat<T>;
                const auto [d,r,alpha] = constants;
                auto M = kinetools::dh::transform<matrix_type>(d,joint.theta,r,alpha);
                return M;
            }
            auto operator()(const std::monostate) const -> Mat<T> {
                assert(false);
                /* throw std::runtime_error{}; */
            }
            const std::array<T,3> constants;
        };

        enum class JointType {
            Prismatic, Rotational
        };

        template <typename T>
        struct DHJoint {
            DHJoint(const std::array<T,3> constants, const JointType type) 
                : constants(constants), type(type) {}
            auto operator()(const T &joint) const {
                Joint<T> j{};
                if (type==JointType::Prismatic)
                    j = Prismatic<T>{joint};
                else j = Rotational<T>{joint};
                return std::visit(JointVisitor{constants},j);
            }
            auto operator()(const Joint<T> &joint) const {
                if (type==JointType::Prismatic)
                    assert(std::holds_alternative<Prismatic<T>>(joint));
                else assert(std::holds_alternative<Rotational<T>>(joint));
                return std::visit(JointVisitor{constants},joint);
            }
            std::array<T,3> constants;
            JointType type;
        };
        
    } // namespace joints
} // namespace kinetools

#endif // KINETOOLS_JOINTS_EIGEN_HPP