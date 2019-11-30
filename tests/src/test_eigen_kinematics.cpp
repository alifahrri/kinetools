#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include "kinetools/kinetools.hpp"

template <typename T>
using Mat = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>;
template <typename T>
using Vec = Eigen::Matrix<T,Eigen::Dynamic,1>;

TEST(dh, transz)
{
    double d{1.0};
    auto M = kinetools::dh::transz<Mat<double>>(d);
    Mat<double> expected(4,4);
    expected << 
        1.0, 0., 0., 0.,
        0., 1.0, 0., 0.,
        0., 0., 1.0, d,
        0., 0., 0., 1.;
    ASSERT_EQ(M.rows(), expected.rows());
    ASSERT_EQ(M.cols(), expected.cols());
    for (size_t i=0; i<M.rows(); i++)
        for (size_t j=0; j<M.cols(); j++)
            EXPECT_NEAR(M(i,j),expected(i,j),1e-12);
}

TEST(dh, transx)
{
    double r{1.0};
    auto M = kinetools::dh::transx<Mat<double>>(r);
    Mat<double> expected(4,4);
    expected << 
        1.0, 0., 0., r,
        0., 1.0, 0., 0.,
        0., 0., 1.0, 0.,
        0., 0., 0., 1.;
    ASSERT_EQ(M.rows(), expected.rows());
    ASSERT_EQ(M.cols(), expected.cols());
    for (size_t i=0; i<M.rows(); i++)
        for (size_t j=0; j<M.cols(); j++)
            EXPECT_NEAR(M(i,j),expected(i,j),1e-12);
}

TEST(dh, rotz)
{
    double t{1.0};
    auto M = kinetools::dh::rotz<Mat<double>>(t);
    Mat<double> expected(4,4);
    expected << 
        cos(t), -sin(t), 0., 0.,
        sin(t), cos(t), 0., 0.,
        0., 0., 1.0, 0.,
        0., 0., 0., 1.;
    ASSERT_EQ(M.rows(), expected.rows());
    ASSERT_EQ(M.cols(), expected.cols());
    for (size_t i=0; i<M.rows(); i++)
        for (size_t j=0; j<M.cols(); j++)
            EXPECT_NEAR(M(i,j),expected(i,j),1e-12);
}

TEST(dh, rotx)
{
    double a{1.0};
    auto M = kinetools::dh::rotx<Mat<double>>(a);
    Mat<double> expected(4,4);
    expected << 
        1.0, 0., 0., 0.,
        0., cos(a), -sin(a), 0.,
        0., sin(a), cos(a), 0.,
        0., 0., 0., 1.;
    ASSERT_EQ(M.rows(), expected.rows());
    ASSERT_EQ(M.cols(), expected.cols());
    for (size_t i=0; i<M.rows(); i++)
        for (size_t j=0; j<M.cols(); j++)
            EXPECT_NEAR(M(i,j),expected(i,j),1e-12);
}

TEST(dh, transform)
{
    double d{1.0};
    double r{1.0};
    double a{0.0};
    double t{1.0};
    auto M = kinetools::dh::transform<Mat<double>>(d,t,r,a);
    Mat<double> expected(4,4);
    expected << 
        cos(t), -sin(t)*cos(a), sin(t)*sin(a), r*cos(t),
        sin(t), cos(t)*cos(a), -cos(t)*sin(a), r*sin(t),
        0., sin(a), cos(a), d,
        0., 0., 0., 1.;
    ASSERT_EQ(M.rows(), expected.rows());
    ASSERT_EQ(M.cols(), expected.cols());
    for (size_t i=0; i<M.rows(); i++)
        for (size_t j=0; j<M.cols(); j++)
            EXPECT_NEAR(M(i,j),expected(i,j),1e-12);
}

TEST(kinetools, rotmat)
{
    double t{1.0};
    auto M = kinetools::dh::rotz<Mat<double>>(t);
    auto R = kinetools::rotmat(M);
    Mat<double> expected(3,3);
    expected << 
        cos(t), -sin(t), 0.,
        sin(t), cos(t), 0., 
        0., 0., 1.0;
    ASSERT_EQ(R.rows(), expected.rows());
    ASSERT_EQ(R.cols(), expected.cols());
    for (size_t i=0; i<R.rows(); i++)
        for (size_t j=0; j<R.cols(); j++)
            EXPECT_NEAR(R(i,j),expected(i,j),1e-12);
}

TEST(kinetools, transmat)
{
    double d{1.0};
    auto M = kinetools::dh::transz<Mat<double>>(d);
    auto T = kinetools::transmat(M);
    Mat<double> expected(3,1);
    expected << 0., 0., d;
    ASSERT_EQ(T.rows(), expected.rows());
    ASSERT_EQ(T.cols(), expected.cols());
    for (size_t i=0; i<T.rows(); i++)
        for (size_t j=0; j<T.cols(); j++)
            EXPECT_NEAR(T(i,j),expected(i,j),1e-12);
}

TEST(kinetools, base_transform)
{
    namespace dh = kinetools::dh;
    namespace traits = kinetools::utility::traits;
    {
        double d{1.0}, t{1.0};
        std::vector<Mat<double>> transforms;
        transforms.push_back(
            dh::transz<Mat<double>>(d)
        );
        transforms.push_back(
            dh::rotz<Mat<double>>(t)
        );
        static_assert(traits::is_resizeable<decltype(transforms)>::value);
        auto tf = kinetools::base_transforms(transforms);
        ASSERT_EQ(tf.size(),2);
        std::vector<Mat<double>> expected;
        expected.push_back(
            dh::transz<Mat<double>>(d)
        );
        expected.push_back(
            dh::transz<Mat<double>>(d) * dh::rotz<Mat<double>>(t)
        );
        ASSERT_EQ(tf.size(),expected.size());
        for (size_t i=0; i<std::size(expected); i++) {
            ASSERT_EQ(tf[i].cols(),expected[i].cols());
            ASSERT_EQ(tf[i].rows(),expected[i].rows());
            for (size_t j=0; j<tf[i].rows(); j++)
                for (size_t k=0; k<tf[i].cols(); k++)
                    EXPECT_NEAR(tf[i](j,k),expected[i](j,k),1e-12) << tf[i] << '\n' << expected[i];
        }
    }
    {
        double d{1.0}, t{1.0};
        std::array<Mat<double>,2> transforms;
        transforms[0] = dh::transz<Mat<double>>(d);
        transforms[1] = dh::rotz<Mat<double>>(t);
        static_assert(!traits::is_resizeable<decltype(transforms)>::value);
        auto tf = kinetools::base_transforms(transforms);
        ASSERT_EQ(tf.size(),2);
        std::vector<Mat<double>> expected;
        expected.push_back(
            dh::transz<Mat<double>>(d)
        );
        expected.push_back(
            dh::transz<Mat<double>>(d) * dh::rotz<Mat<double>>(t)
        );
        ASSERT_EQ(tf.size(),expected.size());
        for (size_t i=0; i<std::size(expected); i++) {
            ASSERT_EQ(tf[i].cols(),expected[i].cols());
            ASSERT_EQ(tf[i].rows(),expected[i].rows());
            for (size_t j=0; j<tf[i].rows(); j++)
                for (size_t k=0; k<tf[i].cols(); k++)
                    EXPECT_NEAR(tf[i](j,k),expected[i](j,k),1e-12) << tf[i] << '\n' << expected[i];
        }
    }
}

TEST(kinetools, rot2omega)
{
    namespace kt = kinetools;
    auto R = Mat<double>(3,3);
    R.setIdentity();
    auto w = kinetools::rotmat_to_omega<Vec<double>>(R);
    EXPECT_EQ(w.size(),3);
}

TEST(kinetools, calc_vw_error)
{
    namespace kt = kinetools;
    auto R_ref = Mat<double>(3,3);
    auto p_ref = Vec<double>(3);
    auto R_now = Mat<double>(3,3);
    auto p_now = Vec<double>(3);
    R_ref.setIdentity();
    p_ref.setZero();
    R_now = R_ref * 2;
    p_now[0] = 1;
    p_now[1] = 1;
    p_now[2] = 1;
    auto [p_err, w_err] = kinetools::calc_vw_error(p_ref, p_now, R_ref, R_now);
    EXPECT_EQ(w_err.size(),3);
    EXPECT_EQ(p_err.size(),3);
    EXPECT_NEAR(std::fabs(p_err[0]),1,1e-6);
    EXPECT_NEAR(std::fabs(p_err[1]),1,1e-6);
    EXPECT_NEAR(std::fabs(p_err[2]),1,1e-6);
}