#ifndef KINETOOLS_HPP
#define KINETOOLS_HPP

#include "kinetools/utility.hpp"

namespace kinetools {

    template <typename M>
    using is_sliceable_matrix = std::conjunction<
        utility::traits::is_callable<M,size_t,size_t>,
        utility::traits::has_cols<M>,
        utility::traits::has_rows<M>
    >;

    namespace dh {

        template <typename M, typename S, typename = void>
        struct is_dh_matrix : std::false_type {};
        template <typename M, typename S>
        struct is_dh_matrix<M, S, std::enable_if_t<
            std::conjunction_v<
                std::disjunction<
                    utility::traits::has_Identity<M>, 
                    utility::traits::has_Identity<M,size_t,size_t>>,
                utility::traits::is_callable<M,size_t,size_t>,
                std::is_assignable<decltype(std::declval<M>()(0,0)),S>
            > >
        > : std::true_type {};

        template <typename Matrix, typename Scalar>
        Matrix transz(const Scalar& dn)
        {
            static_assert(
                is_dh_matrix<Matrix,Scalar>::value,
                "expects Matrix::Identiy(); m(0,0) = Scalar{}; is valid"
            );
            Matrix M;
            if constexpr (utility::traits::has_Identity<Matrix,size_t,size_t>::value) 
                M = Matrix::Identity(4,4);
            else M = Matrix::Identity();
            M(2,3) = dn;
            return M;
        }

        template <typename Matrix, typename Scalar>
        Matrix transx(const Scalar& rn)
        {
            static_assert(
                is_dh_matrix<Matrix,Scalar>::value,
                "expects Matrix::Identiy(); m(0,0) = Scalar{}; is valid"
            );
            Matrix M;
            if constexpr (utility::traits::has_Identity<Matrix,size_t,size_t>::value) 
                M = Matrix::Identity(4,4);
            else M = Matrix::Identity();
            M(0,3) = rn;
            return M;
        }

        template <typename Matrix, typename Scalar>
        Matrix rotz(const Scalar& theta)
        {
            static_assert(
                is_dh_matrix<Matrix,Scalar>::value,
                "expects Matrix::Identiy(); m(0,0) = Scalar{}; is valid"
            );
            Matrix M;
            if constexpr (utility::traits::has_Identity<Matrix,size_t,size_t>::value) 
                M = Matrix::Identity(4,4);
            else M = Matrix::Identity();
            M(0,0) = cos(theta); M(0,1) = -sin(theta);
            M(1,0) = sin(theta); M(1,1) = cos(theta);
            return M;
        }

        template <typename Matrix, typename Scalar>
        Matrix rotx(const Scalar& alpha)
        {
            static_assert(
                is_dh_matrix<Matrix,Scalar>::value,
                "expects Matrix::Identiy(); m(0,0) = Scalar{}; is valid"
            );

            Matrix M;
            if constexpr (utility::traits::has_Identity<Matrix,size_t,size_t>::value) 
                M = Matrix::Identity(4,4);
            else M = Matrix::Identity();
            M(1,1) = cos(alpha); M(1,2) = -sin(alpha);
            M(2,1) = sin(alpha); M(2,2) = cos(alpha);
            return M;
        }

        // template <typename Mat, typename ScalarPos, typename ScalarRot>
        // Mat transform(const ScalarPos &d, const ScalarRot &theta, const ScalarPos &r, const ScalarRot &alpha)
        template <typename Mat>
        Mat transform(const auto &d, const auto &theta, const auto &r, const auto &alpha)
        {
            // static_assert(
            //     std::conjunction_v<
            //         is_dh_matrix<Mat,ScalarPos>,
            //         is_dh_matrix<Mat,ScalarRot>
            //     >,
            //     "expects Matrix::Identiy(); m(0,0) = Scalar{}; is valid"
            // );

            return transz<Mat>(d) * rotz<Mat>(theta) * transx<Mat>(r) * rotx<Mat>(alpha);
        }

        template <typename Mat, typename IterablePos, typename IterableRot>
        Mat forward(const IterablePos &d, const IterableRot &t, const IterablePos &r, const IterableRot &a)
        {
            static_assert(
                std::conjunction_v<
                    utility::traits::is_iterable<IterablePos>,
                    utility::traits::is_iterable<IterableRot>
                >, "expects IterablePos and IterableRot type to be iterable"
            );
            static_assert(
                utility::traits::has_Identity<Mat>::value, 
                "expects Mat::Identity is valid"  
            );
            assert(
                std::size(d)==std::size(a) && 
                std::size(a)==std::size(r) && 
                std::size(r)==std::size(t)
            );

            auto M = Mat::Identity();
            for (size_t i=0; i<std::size(d); i++) 
                M = M * transform<Mat>(d[i],t[i],r[i],a[i]);
            return M;
        }

    } // namespace dh

    template <typename Mat>
    auto rotmat(const Mat &T) 
    {
        static_assert(
            is_sliceable_matrix<Mat>::value, 
            "expects T(0,0), T{}.rows(), T{}.cols() is valid"
        );
        assert(T.rows()==4);
        assert(T.cols()==4);
        return utility::block(T,0,0,3,3);
    }

    template <typename Mat>
    auto transmat(const Mat &T)
    {
        static_assert(
            is_sliceable_matrix<Mat>::value, 
            "expects T(0,0), T{}.rows(), T{}.cols() is valid"
        );
        assert(T.rows()==4);
        assert(T.cols()==4);
        return utility::block(T,0,3,3,1);
    }

    template <typename IterableMat>
    auto base_transform(const IterableMat &transforms, size_t n)
    {
        static_assert(
            std::conjunction_v<
                utility::traits::is_iterable<IterableMat>, utility::traits::is_indexable<IterableMat>,
                utility::traits::has_back<IterableMat>, utility::traits::has_front<IterableMat>
            >, "expect IterableMat type to be iterable, has .back() and .front()!"
        );
        using Mat = std::decay_t<decltype(transforms[0])>;
        static_assert(
            utility::traits::is_multiplicative<Mat,Mat>::value,
            "expects the value type of IterableMat to be multiplicative"
        );
        assert(transforms.front().rows()==4);
        assert(transforms.front().cols()==4);
        assert(n<std::size(transforms));
        auto tf = transforms.front();
        for (size_t i=1; i<=n; i++)
            tf = tf * transforms[i];
        return tf;
    }

    template <typename IterableMat>
    auto base_transforms(const IterableMat &transforms)
    {
        static_assert(
            std::conjunction_v<
                utility::traits::is_iterable<IterableMat>,
                utility::traits::is_indexable<IterableMat>,
                utility::traits::has_back<IterableMat>,
                utility::traits::has_front<IterableMat>
            >, "expect IterableMat type to be iterable, has .back() and .front()!"
        );
        using Mat = std::decay_t<decltype(transforms[0])>;
        static_assert(
            std::conjunction_v<
                kinetools::is_sliceable_matrix<Mat>,
                utility::traits::has_Identity<Mat,size_t,size_t>,
                utility::traits::has_Zero<Mat,size_t,size_t>
            >,
            "expects the value type of IterableMat to be sliceable"
        );
        assert(transforms.front().rows()==4);
        assert(transforms.front().cols()==4);
        auto tfs = IterableMat{};
        if constexpr (utility::traits::is_resizeable<IterableMat>::value) 
            tfs.resize(std::size(transforms));
        for (size_t i=0; i<std::size(transforms); i++)
            tfs[i] = base_transform(transforms,i);
        return tfs;
    }

    namespace constraits {
        namespace tf {
            template <typename T>
            inline constexpr bool is_iterable_v = std::conjunction_v<
                utility::traits::is_iterable<T>, utility::traits::is_indexable<T>,
                utility::traits::has_back<T>, utility::traits::has_front<T>
            >;
        } // namespace tf
    } // namespace constraits

    template <typename IterableMat, typename IterableArgs>
    auto base_transform(IterableMat &transforms, const IterableArgs &args, size_t n)
    {
        static_assert(
            constraits::tf::is_iterable_v<IterableMat> 
            // && constraits::tf::is_iterable_v<IterableArgs>
            , "expect IterableMat and IterableArgs type to be iterable, has .back() and .front()!"
        );
        using Mat = std::decay_t<decltype(transforms[0])>;
        using Arg = std::decay_t<decltype(args[0])>;
        static_assert(
            utility::traits::is_callable<Mat,Arg>::value,
            "expect value type of IterableMat is callable with value type of IterableArgs as argument"
        );
        using Ret = std::decay_t<decltype(std::declval<Mat>()(std::declval<Arg>()))>;
        static_assert(
            utility::traits::is_multiplicative<Ret,Ret>::value,
            "expects the return value of the invocation of value type of IterableMat to be multiplicative"
        );
        assert(std::size(transforms)==std::size(args));
        assert(n<std::size(transforms));
        auto tf = transforms.front()(args[0]);
        for (size_t i=1; i<=n; i++)
            tf = tf * transforms[i](args[i]);
        return tf;
    }

    template <typename IterableMat, typename IterableArgs>
    auto base_transforms(const IterableMat &transforms, const IterableArgs &args)
    {
        static_assert(
            constraits::tf::is_iterable_v<IterableMat> && constraits::tf::is_iterable_v<IterableArgs>,
            "expect IterableMat and IterableArgs type to be iterable, has .back() and .front()!"
        );
        using Mat = std::decay_t<decltype(transforms[0])>;
        using Arg = std::decay_t<decltype(args[0])>;
        static_assert(
            utility::traits::is_callable<Mat,Arg>::value,
            "expect value type of IterableMat is callable with value type of IterableArgs as argument"
        );
        using Ret = std::decay_t<decltype(std::declval<Mat>()(std::declval<Arg>()))>;
        static_assert(
            utility::traits::is_multiplicative<Ret,Ret>::value,
            "expects the return value of the invocation of value type of IterableMat to be multiplicative"
        );
        assert(std::size(transforms)==std::size(args));
        auto tfs = utility::types::copy_std_container_t<IterableMat,Ret>{};
        if constexpr (utility::traits::is_resizeable<IterableMat>::value) 
            tfs.resize(std::size(transforms));
        for (size_t i=0; i<std::size(transforms); i++)
            tfs[i] = base_transform(transforms,args,i);
        return tfs;
    }

    template <typename Vec, typename Mat, typename Scalar=typename Mat::Scalar>
    auto rotmat_to_omega(const Mat &R, const Scalar &eps = Scalar{1e-6})
    {
        static_assert(
            std::disjunction_v<
                std::is_constructible<Vec,size_t>, 
                std::is_constructible<Vec>
            >, "expects Vec to be constructible with size_t{} or constructible and resizeable"
        );
        static_assert(
            std::conjunction_v<
                is_sliceable_matrix<Mat>,
                utility::traits::has_norm<Mat>
            >, "expects M(0,0), M.rows(), M.cols() is valid and M has norm()"
        );
        auto op = [&](auto &e, auto &w){
            e[0] = R(2,1) - R(1,2);
            e[1] = R(0,2) - R(2,0);
            e[2] = R(1,0) - R(0,1);
            auto norm = e.norm();
            if (norm > eps) {
                w = atan2(norm, R.trace()-1) / norm * e;
            } else if (R(0,0)>0 && R(1,1)>0 && R(2,2)>0) {
                w[0] = 0;
                w[1] = 0;
                w[2] = 0;
            } else {
                w[0] = R(0,0)+1;
                w[1] = R(1,1)+1;
                w[2] = R(2,2)+1;
                w = M_PI / 2 *w;
            }
        };
        if constexpr (std::is_constructible_v<Vec,size_t>) {
            Vec e(3); Vec w(3);
            op(e,w);
            return w;
        } else {
            Vec e; Vec w;
            op(e,w);
            return w;
        }
    }

    template <typename PVec, typename PNVec, typename RMat, typename RNMat>
    auto calc_vw_error(const PVec &p_ref, const PNVec &p_now, const RMat &R_ref, const RNMat &R_now)
    {
        static_assert(
            std::disjunction_v<
                std::is_constructible<PVec,size_t>, 
                std::is_constructible<PVec>
            >, "expects Vec to be constructible with size_t{} or constructible and resizeable"
        );
        static_assert(
            std::disjunction_v<
                utility::traits::has_inverse<RMat>,
                utility::traits::has_inv<RMat>
            >, "expects RMat to have inverse() or inv()"
        );
        auto p_err = p_ref - p_now;
        auto inv = [&](const RMat &R) {
            if constexpr (utility::traits::has_inverse<RMat>::value)
                return R.inverse();
            else if constexpr (utility::traits::has_inv<RMat>::value)
                return R.inv();
        };
        auto R_err = inv(R_now) * R_ref;
        auto w_err = R_now * rotmat_to_omega<PVec>(R_err);
        return std::make_tuple(p_err, w_err);
    }

    template <typename JMat, typename WMat, typename WnMat, typename Vec, typename Scalar>
    auto compute_dq(const JMat &J, const WMat &We, const WnMat &Wn, const Vec &err, Scalar Ek)
    {
        static_assert(
            std::conjunction_v<
                utility::traits::has_transpose<JMat>,
                utility::traits::has_transpose<Vec>
            >, "expects JMat and Vec instances to have .transpose()"
        );
        JMat Jh = J.transpose() * We * J + Wn * (Ek + 2e-3);

        static_assert(
            utility::traits::has_colPivHouseholderQr<std::decay_t<decltype(Jh)>>::value
            , "expects the result of J.transpose() * We * J to have .colPivHouseholderQr() !"
        );

        Vec g_err = J.transpose() * We * err;
        auto solver = Jh.colPivHouseholderQr();
        // auto solver = Jh.householderQr();

        static_assert(
            utility::traits::has_solve<std::decay_t<decltype(solver)>,std::decay_t<decltype(g_err)>>::value
            , "expects the result of (J.transpose() * We * J).colPivHouseholderQr() to have solve() !"
        );

        assert(g_err.cols()==1);
        assert(g_err.rows()==Jh.rows());
        Vec dq = solver.solve(g_err);
        // auto dq = solver.solve(g_err);
        // Vec dq = (Jh.transpose() * Jh).ldlt().solve(Jh.transpose() * g_err);
        return dq;
    }

    template <typename Vec, typename Mat, typename Forward, typename Jacobian, typename Joints>
    auto inverse_kinematic_lm(const Vec &p_ref, const Mat &R_ref, Forward &forward, Jacobian &jacobian, const Joints &q_init, size_t iter=50)
    {
        static_assert(
            std::conjunction_v<
                std::is_constructible<Vec,size_t>
                , utility::traits::has_asDiagonal<Vec>
                // , utility::traits::has_comma_initializer<Vec,typename Vec::Scalar>
            >, "expects Vec to have .asDiagonal() and Vec() << Vec::Scalar{}, ...; is valid"
        );
        static_assert(
            std::conjunction_v<
                utility::traits::has_Identity<Mat,size_t,size_t>
            >, "expects Mat to have ::Identity(size_t{},size_t{})"
        );

        using Scalar = typename Vec::Scalar;

        auto get_transform_mat_tuple = [](const auto &T) {
            using TMat = std::decay_t<decltype(T)>;
            static_assert(
                is_sliceable_matrix<TMat>::value
            );
            if constexpr (is_sliceable_matrix<TMat>::value) {
                assert(T.rows()==4);
                assert(T.cols()==4);
                return std::make_tuple(
                    rotmat(T), transmat(T)
                );
            } else return T;
        };

        auto wn_pos = 1./0.3; auto wn_ang = 1./(2*M_PI);

        Vec ve(6);
        ve << wn_pos, wn_pos, wn_pos, wn_ang, wn_ang, wn_ang;

        auto We = ve.asDiagonal();
        auto Wn = Mat::Identity(q_init.rows(),q_init.rows());

        auto T = forward(q_init);
        auto [R_now, p_now] = get_transform_mat_tuple(T);

        Vec p_err(3);
        Vec w_err(3);
        std::tie(p_err, w_err) = calc_vw_error(p_ref,p_now,R_ref,R_now);

        assert(p_err.size()==3);
        assert(w_err.size()==3);

        auto err = Vec(6); 
        err << p_err[0], p_err[1], p_err[2], w_err[0], w_err[1], w_err[2];

        auto q_now = q_init;
        auto Ek = static_cast<Scalar>(err.transpose() * We * err);
        for (size_t i=0; i<iter; i++) {
            auto J = jacobian(q_now);
            auto dq = compute_dq(J,We,Wn,err,Ek);
            q_now = q_now + dq;
            T = forward(q_now);
            auto [R_now, p_now] = get_transform_mat_tuple(T);

            std::tie(p_err,w_err) = calc_vw_error(p_ref,p_now,R_ref,R_now);
            assert(p_err.size()==3);
            assert(w_err.size()==3);
            err << p_err[0], p_err[1], p_err[2], w_err[0], w_err[1], w_err[2];
            auto Ek2 = static_cast<Scalar>(err.transpose() * We * err);
            if (Ek2 < 1e-12)
                break;
            else if (Ek2 < Ek)
                Ek = Ek2;
            else {
                q_now = q_now - dq;
                T = forward(q_init);
            }
        }
        return std::make_tuple(q_now,Ek);
    }

} // namespace kinematics

#endif // KINETOOLS_HPP