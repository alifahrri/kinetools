# kinetools [![Build Status](https://travis-ci.com/alifahrri/kinetools.svg?branch=master)](https://travis-ci.com/alifahrri/kinetools)
Jacobian Inverse Kinematic using Automatic Differentiation

## Examples
```C++
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>
#include "kinetools/kinetools.hpp"
#include "kinetools/joints/eigen.hpp"
#include "kinetools/models/eigen/scara.hpp"
#include <iostream>

int main(int argc, char**argv)
{
    namespace kt = kinetools;
    using Vec = kt::joints::Vec<double>;
    using Mat = kt::joints::Mat<double>;
    using Joints = Vec;
    
    /* create scara model given DH parameters */
    auto scara = kt::Scara<double>({
        std::array<double,3>{1.0,1.0,0.0},
        std::array<double,3>{0.0,1.0,M_PI},
        std::array<double,3>{0.0,0.0,0.0},
        std::array<double,3>{1.0,0.0,0.0}
    });

    auto q_init = Vec(4);
    for (size_t i=0; i<5; i++) {
        auto p_ref = Vec(3);
        auto R_ref = Mat(3,3);
        p_ref << 1.0, (i*1./5.)+0.1, 0.0;
        R_ref.setIdentity();
        auto [q, err] = scara.inverse(p_ref,R_ref,q_init);
        auto q_array = std::array<double,4>{
            q[0], q[1], q[2], q[3]
        };
        auto tf = scara.forward(q_array);
        std::cout << "target : " << p_ref.transpose() << std::endl;
        std::cout << "q : " << q.transpose() << std::endl;
        std::cout << "err : " << err << std::endl;
        std::cout << "tf : " << tf.back() << std::endl;
        q_init = q;
    }
}
```

## Requirements
```
C++17
```
```
build-essential
git
wget
cmake
pkg-config
tar
gdb
python3
python3-pip
```
Additionally, you'll need these libraries (see docker/dockerfile for installing) :
```
gtest
Eigen
CppAD
```
## How To
check docker/dockerfile

## Directory structure
```
  | docker                      ## dockerfile and helper scripts
  | examples                    ## examples
  | include
      | kinetools
          | jacobian
              | cppad.hpp       ## jacobian with CppAD backend
          | joints
              | eigen.hpp       ## joint model with eigen backend
          | models
              | eigen
                  | scara.hpp   ## scara impl
          | jacobian.hpp
          | kinetools.hpp       ## forward and inverse
          | utility.hpp         ## utility and traits
  | CMakeLists.txt
```
