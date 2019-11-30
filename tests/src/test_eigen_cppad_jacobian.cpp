#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>
#include <cppad/example/cppad_eigen.hpp>
#include "kinetools/jacobian.hpp"
#include "kinetools/kinetools.hpp"
#include "kinetools/joints/eigen.hpp"
#include "kinetools/models/eigen/scara.hpp"

TEST(kinetools, cppad_translation_jacobian)
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
    {
        CppAD::vector<double> x(std::size(tfs));
        auto theta_1 = M_PI/4;
        auto theta_2 = M_PI/4;
        auto d_3 = 1.25;
        auto theta_4 = M_PI/4;
        x[0] = theta_1;
        x[1] = theta_2;
        x[2] = d_3;
        x[3] = theta_4;
        auto J = F.Jacobian(x);
        std::vector<double> expected{
            -1.0*sin(theta_1) - 1.0*sin(theta_1 + theta_2), -1.0*sin(theta_1 + theta_2), 0, 0, 
            1.0*cos(theta_1) + 1.0*cos(theta_1 + theta_2), 1.0*cos(theta_1 + theta_2), 0, 0, 
            0, 0, -1, 0, 
            0, 0, 0, 0
        };
        ASSERT_EQ(J.size(),expected.size());
        for (size_t i=0; i<J.size(); i++) 
            EXPECT_NEAR(J[i],expected[i],1e-6);
    }
}

TEST(kinetools, cppad_rotational_jacobian)
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
        auto M = Jw(x);
        std::vector<std::vector<double>> expected{
            std::vector<double>{0, 0, 1},
            std::vector<double>{0, 0, -1},
            std::vector<double>{0, 0, 0},
            std::vector<double>{0, 0, -1}
        };
        ASSERT_EQ(M.size(),expected.size());
        for (size_t i=0; i<M.size(); i++) {
            ASSERT_EQ(M[i].rows(),expected[i].size());
            for (size_t j=0; j<M[i].rows(); j++)
                EXPECT_NEAR(CppAD::Value(M[i][j]),expected[i][j],1e-6);
        }
    }
}

TEST(kinetools, scara_forward)
{
    namespace kt = kinetools;
    auto scara = kt::Scara<double>({
        std::array<double,3>{1.0,1.0,0.0},
        std::array<double,3>{0.0,1.0,M_PI},
        std::array<double,3>{0.0,0.0,0.0},
        std::array<double,3>{1.0,0.0,0.0}
    });
    auto base_tfs = scara.forward({0.,0.,0.,0.});
}

TEST(kinetools, scara_jacobian)
{
    namespace kt = kinetools;
    auto scara = kt::Scara<double>({
        std::array<double,3>{1.0,1.0,0.0},
        std::array<double,3>{0.0,1.0,M_PI},
        std::array<double,3>{0.0,0.0,0.0},
        std::array<double,3>{1.0,0.0,0.0}
    });
    auto theta_1 = M_PI/4;
    auto theta_2 = M_PI/4;
    auto d_3 = 1.25;
    auto theta_4 = M_PI/4;
    auto J = scara.jacobian({theta_1,theta_2,d_3,theta_4});
    auto expected = kt::joints::Mat<double>(6,4);
    expected << -1.0*sin(theta_1) - 1.0*sin(theta_1 + theta_2), -1.0*sin(theta_1 + theta_2), 0, 0, 
            1.0*cos(theta_1) + 1.0*cos(theta_1 + theta_2), 1.0*cos(theta_1 + theta_2), 0, 0, 
            0, 0, -1, 0, 
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, -1, 0, -1;
    ASSERT_EQ(J.rows(),expected.rows());
    ASSERT_EQ(J.cols(),expected.cols());
    for (size_t i=0; i<J.rows(); i++)
        for (size_t j=0; j<J.cols(); j++)
            EXPECT_NEAR(J(i,j),expected(i,j),1e-6) << "(" << i << "," << j << ")";
}

TEST(kinetools, compute_dq)
{
    namespace kt = kinetools;
    auto scara = kt::Scara<double>({
        std::array<double,3>{1.0,1.0,0.0},
        std::array<double,3>{0.0,1.0,M_PI},
        std::array<double,3>{0.0,0.0,0.0},
        std::array<double,3>{1.0,0.0,0.0}
    });
    auto theta_1 = M_PI/4;
    auto theta_2 = M_PI/4;
    auto d_3 = 1.25;
    auto theta_4 = M_PI/4;
    auto J = scara.jacobian({theta_1,theta_2,d_3,theta_4});
    using Vec = kt::joints::Vec<double>;
    using Mat = kt::joints::Mat<double>;
    auto wn_pos = 1./0.3; auto wn_ang = 1./(2*M_PI);
    Vec ve(6);
    ve << wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang;
    auto err = Vec(6);
    auto We = ve.asDiagonal();
    auto Wn = Mat::Identity(4,4);
    auto dq = kt::compute_dq(J,We,Wn,err,0.1);
    // auto Jh = J.transpose() * We * J + Wn;
    // ASSERT_EQ(Jh.rows(),4);
    // ASSERT_EQ(Jh.cols(),4);
    // auto g_err = J.transpose() * We * err;
    // ASSERT_EQ(g_err.rows(),4);
    // ASSERT_EQ(g_err.cols(),1);
    // auto dq = Jh.householderQr().solve(g_err);
    ASSERT_EQ(dq.rows(),4);
    ASSERT_EQ(dq.cols(),1);
}

TEST(kinetools, inverse_kinematic_lm)
{
    namespace kt = kinetools;
    using Vec = kt::joints::Vec<double>;
    using Mat = kt::joints::Mat<double>;
    using Joints = Vec;
    
    auto scara = kt::Scara<double>({
        std::array<double,3>{1.0,1.0,0.0},
        std::array<double,3>{0.0,1.0,M_PI},
        std::array<double,3>{0.0,0.0,0.0},
        std::array<double,3>{1.0,0.0,0.0}
    });

    auto p_ref = Vec(3);
    auto R_ref = Mat(3,3);
    p_ref << 1.0, 0.3, 0.0;
    R_ref.setIdentity();
    auto fwd = [&](const Joints &q){
        std::array<double,4> q_array{
            q[0], q[1], q[2], q[3]
        };
        auto transforms = scara.forward(q_array);
        auto Tb = transforms.back();
        auto T = Mat(4,4);
        for (size_t i=0; i<4; i++)
            for (size_t j=0; j<4; j++)
                T(i,j) = CppAD::Value(Tb(i,j));
        return T;
    };
    auto jac = [&](const Joints &q){
        std::array<double,4> q_array{
            q[0], q[1], q[2], q[3]
        };
        return scara.jacobian(q_array);
    };
    auto q_init = Vec(4);
    auto [q,err] = kt::inverse_kinematic_lm(p_ref,R_ref,fwd,jac,q_init);
    EXPECT_NEAR(err,0.,1e-3);
}

TEST(kinetools, scara_inverse)
{
    namespace kt = kinetools;
    using Vec = kt::joints::Vec<double>;
    using Mat = kt::joints::Mat<double>;
    using Joints = Vec;
    
    auto scara = kt::Scara<double>({
        std::array<double,3>{1.0,1.0,0.0},
        std::array<double,3>{0.0,1.0,M_PI},
        std::array<double,3>{0.0,0.0,0.0},
        std::array<double,3>{1.0,0.0,0.0}
    });

    auto p_ref = Vec(3);
    auto R_ref = Mat(3,3);
    p_ref << 1.0, 0.5, 0.0;
    R_ref.setIdentity();
    auto q_init = Vec(4);
    auto [q, err] = scara.inverse(p_ref,R_ref,q_init);
    EXPECT_NEAR(err,0.,1e-3);
    auto q_array = std::array<double,4>{
        q[0], q[1], q[2], q[3]
    };
    auto tfs = scara.forward(q_array);
    Vec expected(3);
    expected << 1.0, 0.5, 0.0;
    auto tf = kt::transmat(tfs.back());
    ASSERT_EQ(tf.rows(),expected.rows());
    ASSERT_EQ(tf.cols(),expected.cols());
    for (size_t i=0; i<tf.rows(); i++)
        EXPECT_NEAR(CppAD::Value(tf(i,0)),expected[i],1e-6);
}