#include <cppad/cppad.hpp>
#include <iostream>

namespace fun {

    template <typename Type>
    auto func1(const Type &x1, const Type &x2)
    {
        Type v1 = x;
        Type v2 = Type{1} + v1;
        Type v3 = v1 * v1;
        Type v4 = v3 / Type{2};
        Type v5 = v2 + v4;
        return v5;
    }

} // namespace fun