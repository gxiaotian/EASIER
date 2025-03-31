#ifndef ESR_UTILS_H
#define ESR_UTILS_H

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <cxxabi.h>


template <typename T>
std::string get_type_name() {
  const char* name = typeid(T).name();
  int status = -1;

  // Use abi::__cxa_demangle to convert mangled name to readable name
  std::unique_ptr<char, void(*)(void*)> res {
    abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};

  return (status == 0) ? res.get() : name;
}


template <size_t start, size_t end, class Func>
constexpr void constexpr_for(Func&& f) {
  if constexpr (start < end) {
    f.template operator()<start>(),
    constexpr_for<start+1, end>(std::forward<Func>(f));
  }
}


template <size_t start, size_t end, class T, class Func>
consteval auto constexpr_scan(T&& prefix, Func&& f) {
  if constexpr (start < end) {
    return constexpr_scan<start+1, end>(
      f.template operator()<start>(std::forward<T>(prefix)),
      std::forward<Func>(f));
  } else {
    return std::forward<T>(prefix);
  }
}


// template <class... T0, class... T1>
// auto cat_tuples(std::tuple<T0...>& t0, std::tuple<T1...>& t1) {
//   std::cout << "here" << std::endl;
//   auto tmp = std::move(std::tuple<T0..., T1...>{
//     std::get<T0...>(t0),
//     std::get<T1...>(t1)});
//
//   std::get<0>(tmp)[0] = 100;
//   return tmp;
// }


template <class T, class Tuple>
consteval size_t find() {
  constexpr size_t size = std::tuple_size_v<Tuple>;

  constexpr auto ret = constexpr_scan<1, size+1>(0,
    []<size_t I>(auto prefix) {
      using type = std::tuple_element_t<I-1, Tuple>;
      if constexpr (std::is_same_v<type, T>) {
        return I;
      } else {
        return prefix;
      }
    }
  );

  static_assert(ret > 0, "Type T is not found in Tuple.");

  return ret - 1;
}

#endif