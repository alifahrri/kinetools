#ifndef KINETOOLS_UTILITY_HPP
#define KINETOOLS_UTILITY_HPP

#include <type_traits>
#include <array>
#include <vector>
#include <tuple>

namespace kinetools {

    namespace utility {
        namespace traits {

            template <typename T, typename = void>
            struct is_std_array : std::false_type {};
            template <typename T>
            struct is_std_array<T,std::enable_if_t<
                std::is_same_v<std::array<typename T::value_type, std::tuple_size<T>::value>, T>
            > > : std::true_type {};

            template <typename T, typename = void>
            struct is_std_vector : std::false_type {};
            template <typename T>
            struct is_std_vector<T,std::enable_if_t<
                std::is_same_v<std::vector<typename T::value_type>, T>
            > > : std::true_type {};


            template <typename T, typename ...Args>
            struct is_callable {
            private:
                template <typename F>
                static constexpr auto test(int) 
                    -> decltype(std::declval<F>()(std::declval<Args>()...), bool())
                { return true; }
                template <typename F>
                static constexpr auto test(char) 
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            /* TODO : move (?) */
            using std::begin;
            using std::end;

            template <typename T>
            struct is_iterable {
            private:
                template <typename It>
                static constexpr auto test(int)
                    -> decltype(begin(std::declval<It>()), end(std::declval<It>()), bool())
                { return true; }
                template <typename It>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename size_type = size_t, typename ...Args>
            struct is_resizeable {
            private:
                template <typename U>
                static constexpr auto test(int)
                    -> decltype(std::declval<U>().resize(std::declval<size_type>(), std::declval<Args>()...),bool())
                { return true; }
                template <typename U>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename = void>
            struct is_indexable : std::false_type {};
            template <typename T>
            struct is_indexable<T, std::void_t<decltype(std::declval<T>()[size_t{}])> > : std::true_type {};

            template <typename ...Ts>
            using all_iterable = std::conjunction<is_iterable<Ts>...>;
            template <typename ...Ts>
            using all_indexable = std::conjunction<is_indexable<Ts>...>;

            template <typename T, typename U>
            struct is_multiplicative {
            private:
                template <typename A, typename B>
                static constexpr auto test(int) -> decltype(std::declval<A>()*std::declval<B>(),bool())
                { return true; }
                template <typename A, typename B>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T,U>(int{});
            };

            template <typename T, typename = void>
            struct has_value_type : std::false_type {};
            template <typename T>
            struct has_value_type<T, std::void_t<
                typename T::value_type
            > > : std::true_type {};

            template <typename T, typename = void>
            struct has_front : std::false_type {};
            template <typename T>
            struct has_front<T, std::void_t<
                decltype(std::declval<T>().front())
            > > : std::true_type {};

            template <typename T, typename = void>
            struct has_back : std::false_type {};
            template <typename T>
            struct has_back<T, std::void_t<
                decltype(std::declval<T>().back())
            > > : std::true_type {};

            template <typename T, typename = void>
            struct has_push_back : std::false_type {};
            template <typename T>
            struct has_push_back<T, 
                std::void_t<decltype(std::declval<T>().push_back(std::declval<typename T::value_type>()))> >
            : std::true_type {};

            template <typename T, typename = void>
            struct has_resize : std::false_type {};
            template <typename T>
            struct has_resize<T, 
                std::void_t<decltype(std::declval<T>().resize(std::declval<typename T::size_type>()))> >
            : std::true_type {};

            template <typename T, typename = void>
            struct has_resize2d : std::false_type {};
            template <typename T>
            struct has_resize2d<T, 
                std::void_t<decltype(std::declval<T>().resize(size_t{},size_t{}))> >
            : std::true_type {};

            /* TODO : move to matrix traits */

            template <typename T, typename = void>
            struct has_rows : std::false_type {};
            template <typename T>
            struct has_rows<T, 
                std::void_t<decltype(std::declval<T>().rows())> >
            : std::true_type {};

            template <typename T, typename = void>
            struct has_cols : std::false_type {};
            template <typename T>
            struct has_cols<T, 
                std::void_t<decltype(std::declval<T>().cols())> >
            : std::true_type {};

            template <typename T, typename ...Args>
            struct has_Identity {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(M::Identity(std::declval<Args>()...), bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_Zero {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(M::Zero(std::declval<Args>()...), bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_block {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().block(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_norm {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().norm(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_inverse {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().inverse(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_inv {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().inv(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_trace {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().trace(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_transpose {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().transpose(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_solve {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().solve(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_asDiagonal {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().asDiagonal(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename V, typename ...Args>
            struct has_comma_initializer {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().operator<<(std::declval<V>()).operator,(std::declval<Args>()...))
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_bdcSvd {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().bdcSvd(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename ...Args>
            struct has_colPivHouseholderQr {
            private:
                template <typename M>
                static constexpr auto test(int)
                    -> decltype(std::declval<M>().colPivHouseholderQr(std::declval<Args>()...),bool())
                { return true; }
                template <typename M>
                static constexpr auto test(char)
                { return false; }
            public:
                constexpr static bool value = test<T>(int());
            };

            template <typename T, typename = void>
            struct has_setIdentity : std::false_type {};
            template <typename T>
            struct has_setIdentity<T, 
                std::void_t<decltype(std::declval<T>().setIdentity())> >
            : std::true_type {};

            template <typename T, typename = void>
            struct has_setZero : std::false_type {};
            template <typename T>
            struct has_setZero<T, 
                std::void_t<decltype(std::declval<T>().setZero())> >
            : std::true_type {};

            template <typename T>
            using is_std_array_or_vector = std::disjunction<is_std_array<std::decay_t<T>>,is_std_vector<std::decay_t<T>>>;
            
        } // namespace traits

        namespace types {
            /* TODO : make generic container (via template template parameter{?}) */
            template <typename T, typename V, size_t, typename=void>
            struct copy_std_container {
                using type = std::enable_if_t<
                    traits::is_std_array_or_vector<T>::value, std::vector<std::decay_t<V>>
                >;
            };
            template <typename T, typename V>
            struct copy_std_container<T,V,0,std::enable_if_t<traits::is_std_array<T>::value>> {
                using type = std::enable_if_t<
                    traits::is_std_array_or_vector<T>::value, std::array<std::decay_t<V>,std::tuple_size<T>::value>
                >;
            };
            template <typename T, typename V, size_t n>
            struct copy_std_container<T,V,n,std::enable_if_t<traits::is_std_array<T>::value>> {
                using type = std::enable_if_t<
                    traits::is_std_array_or_vector<T>::value, std::array<std::decay_t<V>,n>
                >;
            };
            template <typename T, typename V, size_t n=0>
            using copy_std_container_t = typename copy_std_container<T,V,n>::type;
        } // namespace types

        template <typename Mat, typename Vec>
        auto mul(const Mat& M, const Vec& v) {
            static_assert(
                std::conjunction_v<
                    traits::has_cols<Mat>,
                    traits::has_rows<Mat>,
                    traits::is_indexable<Vec>,
                    traits::is_callable<Mat,size_t,size_t>
                >, "expects Mat to have rows() & cols()"
            );
            Vec result(v.rows());
            for (size_t i=0; i<result.rows(); i++) {
                result[i] = 0.;
                for (size_t j=0; j<v.rows(); j++)
                    result[i] = result[i] + M(i,j) * v[j];
            }
            return result;
        }

        template <typename Mat, typename ...Args>
        auto block(const Mat& M, Args...args)
        {
            static_assert(
                traits::has_block<Mat,Args...>::value,
                "expects Mat type has block member function"
            );
            return M.block(args...);
        }
        template <typename Mat, typename ...Args>
        auto block(Mat& M, Args...args)
        {
            static_assert(
                traits::has_block<Mat,Args...>::value,
                "expects Mat type has block member function"
            );
            return M.block(args...);
        }
    } // namespace utility
} // namespace kinetools

#endif // KINETOOLS_UTILITY_HPP
