#include "kinetools/models/eigen/scara.hpp"
#include <iostream>

int main(int argc, char**argv)
{
    auto scara = kinetools::Scara<double>({
        std::array<double,3>{1.0,1.0,0.0},
        std::array<double,3>{0.0,1.0,M_PI},
        std::array<double,3>{0.0,0.0,0.0},
        std::array<double,3>{1.0,0.0,0.0}
    });
    auto base_tfs = scara.forward({0.,0.,0.,0.});
    std::cout << "base transform :" << std::endl;
    for (size_t i=0; i<base_tfs.size(); i++)
        std::cout << "i(" << i << ") : " << base_tfs[i] << std::endl;
    return 0;
}