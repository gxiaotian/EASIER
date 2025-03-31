#include "binding.hpp"

FORWARD(foo, double, int, double, int, double)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("foo", foo_forward);
}
