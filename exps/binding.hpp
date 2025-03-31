#ifndef ESR_BINDING_H
#define ESR_BINDING_H

#include <torch/extension.h>
#include "utils.hpp"


#define CONCAT(a, b) a##b

#define FORWARD(name, ...) \
  extern void (*name)(Pointers<__VA_ARGS__>); \
  void CONCAT(name, _forward)(Tensors<__VA_ARGS__> tensors) { \
    name(get_ptrs<__VA_ARGS__>(tensors)); \
  }


template<class T>
using Tensor = torch::Tensor;

template<class... Ts>
using Tensors = std::tuple<Tensor<Ts>...>;

template<class T>
using Pointer = std::add_pointer_t<T>;

template<class... Ts>
using Pointers = std::tuple<Pointer<Ts>...>;


template<class... Ts>
inline Pointers<Ts...> get_ptrs(Tensors<Ts...> tensors) {
  Pointers<Ts...> ret{};

  constexpr_for<0, sizeof...(Ts)>(
    [&tensors, &ret]<size_t I>() {
      using T = std::tuple_element_t<I, std::tuple<Ts...>>;
      auto tensor = std::get<I>(tensors);
      std::get<I>(ret) = tensor.template data_ptr<T>();
    }
  );

  return ret;
}

#endif
