#include <cstddef>
#include <tuple>

#include "ops.hpp"
#include "openmp.hpp"

using _select = Node<double, Select<0, 1>>;
using _nnz = Node<double, Load<2>>;
using _mul = Node<double, Mul, _select, _nnz>;
using _reduce = Node<double, Reduce<4, 3>, _mul>;
using Nodes = std::tuple<_select, _nnz, _mul, _reduce>;


void (*foo)(std::tuple<double*, int*, double*, int*, double*>&) =
  // reduce<37464962, 381689, 16, Nodes, std::tuple<double*, int*, double*, int*, double*>>;
  reduce<283073458, 2017169, 4, Nodes, std::tuple<double*, int*, double*, int*, double*>>;
  // all_reduce<11, 2, Nodes, std::tuple<float*, int*, float*, float*>>;
