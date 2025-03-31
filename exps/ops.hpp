#ifndef ESR_OPS_H
#define ESR_OPS_H

#include <cstddef>
#include <tuple>
#include "utils.hpp"


template<class _RetT, class _Op, class... _Args>
struct Node {
  using Op = _Op;
  using Args = std::tuple<_Args...>;
  using RetT = _RetT;
};


/*
 *  Definition of Map operations
 */


template<size_t I>
struct Load {
  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void run(Ptrs& ptrs, Buffs& buffers, const size_t idx) {
    constexpr auto N = find<Node, Nodes>();
    std::get<N>(buffers) = std::get<I>(ptrs)[idx];
  }
};


// I0 is the input data pointer index, and I1 is the index pointer index
template<size_t I0, size_t I1>
struct Select {
  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void run(Ptrs& ptrs, Buffs& buffers, const size_t idx) {
    auto data = std::get<I0>(ptrs);
    auto select_idx = std::get<I1>(ptrs)[idx];

    constexpr auto N = find<Node, Nodes>();
    std::get<N>(buffers) = data[select_idx];
  }
};


struct Add {
  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void run(Ptrs& ptrs, Buffs& buffers, const size_t idx) {
    constexpr auto R = find<Node, Nodes>();

    using Arg0 = std::tuple_element_t<0, typename Node::Args>;
    constexpr auto N0 = find<Arg0, Nodes>();

    using Arg1 = std::tuple_element_t<1, typename Node::Args>;
    constexpr auto N1 = find<Arg1, Nodes>();

    std::get<R>(buffers) = std::get<N0>(buffers) + std::get<N1>(buffers);
  }
};


struct Mul {
  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void run(Ptrs& ptrs, Buffs& buffers, const size_t idx) {
    constexpr auto R = find<Node, Nodes>();

    using Arg0 = std::tuple_element_t<0, typename Node::Args>;
    constexpr auto N0 = find<Arg0, Nodes>();

    using Arg1 = std::tuple_element_t<1, typename Node::Args>;
    constexpr auto N1 = find<Arg1, Nodes>();

    std::get<R>(buffers) = std::get<N0>(buffers) * std::get<N1>(buffers);
  }
};


template<size_t I>
struct Dump {
  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void run(Ptrs& ptrs, Buffs& buffers, const size_t idx) {
    using Arg0 = std::tuple_element_t<0, typename Node::Args>;
    constexpr auto N = find<Arg0, Nodes>();
    std::get<I>(ptrs)[idx] = std::get<N>(buffers);
  }
};


/*
 *  Definition of Reduce operations
 */


struct ReduceOp {};


// I0 is the output data pointer index, and I1 is the index pointer index
template<size_t I0, size_t I1>
struct Reduce : ReduceOp {

  template<class Ptrs>
  inline auto get_row_end_offsets(Ptrs& ptrs) {
    return std::get<I1>(ptrs);
  }

  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void run(Ptrs& ptrs, Buffs& buffers, const size_t idx) {
    using Arg0 = std::tuple_element_t<0, typename Node::Args>;
    constexpr auto N0 = find<Arg0, Nodes>();
    constexpr auto R = find<Node, Nodes>();
    std::get<R>(buffers) += std::get<N0>(buffers);
  }

  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void dump(Ptrs& ptrs, Buffs& buffers, const size_t row) {
    constexpr auto R = find<Node, Nodes>();
    std::get<I0>(ptrs)[row] = std::get<R>(buffers);
  }

  template<class Ptrs, class T>
  inline void dump(Ptrs& ptrs, const T& value, const size_t row) {
    std::get<I0>(ptrs)[row] += value;
  }
};


/*
 *  Definition of AllReduce operations
 */


struct AllReduceOp {};


template<size_t I>
struct Sum : AllReduceOp {

  template<class Node, class Nodes, class Ptrs, class Buffs>
  inline void run(Ptrs& ptrs, Buffs& buffers, const size_t idx) {
    using Arg0 = std::tuple_element_t<0, typename Node::Args>;
    constexpr auto N = find<Arg0, Nodes>();
    constexpr auto R = find<Node, Nodes>();
    std::get<R>(buffers) = std::get<N>(buffers);
  }

  template<
      class Node,
      class Nodes,
      class AllReduceNodes,
      class Buffs,
      class AccBuffs>
  inline void acc(Buffs& buffs, AccBuffs& acc_buffs) {
    constexpr auto i_buf = find<Node, Nodes>();
    constexpr auto i_acc = find<Node, AllReduceNodes>();
    std::get<i_acc>(acc_buffs) += std::get<i_buf>(buffs);
  }

  template<
      class Node,
      class AllReduceNodes,
      class Ptrs,
      class AccBuffs>
  inline void dump(Ptrs& ptrs, AccBuffs& acc_buffs) {
    constexpr auto i_acc = find<Node, AllReduceNodes>();
    *std::get<I>(ptrs) += std::get<i_acc>(acc_buffs);
  }
};



#endif