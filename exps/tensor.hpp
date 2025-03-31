#ifndef ESR_TENSOR_H
#define ESR_TENSOR_H

#include <array>
#include <cstddef>
#include "utils.hpp"


template<size_t start_, size_t end_, size_t step_>
struct Slice {
  constexpr static size_t start = start_;
  constexpr static size_t end = end_;
  constexpr static size_t step = step_;

  constexpr static size_t dim = (end - start) / step;

  consteval auto iter_space() {
    return constexpr_scan<0, dim>(std::tuple(),
      [](auto i, auto prefix) {
        return std::tuple_cat(prefix, std::tuple(start + i.value * step));
      }
    );
  }
};


template<class T, size_t... dims>
struct Tensor {
  std::array<T, (dims * ...)> data;

  consteval size_t size() {
    return (dims * ...);
  }

  consteval size_t dim() {
    return sizeof...(dims);
  }
};

#endif
