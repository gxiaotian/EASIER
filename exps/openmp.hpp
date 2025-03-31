#ifndef ESR_OPENMP_H
#define ESR_OPENMP_H

// Note: Intel MKL compiler is highly recommended, becuase it is usually much
// better than GCC when dealing with openmp programs.

// MKL_DYNAMIC=FALSE OMP/MKL_NUM_THREADS=xxx
// will control number of threads in both PyTorch and EASIER

#include <cstddef>
#include <algorithm>
#include <type_traits>

#include "ops.hpp"
#include "utils.hpp"
// #include "omp.h"


template<class Nodes>
inline auto create_buffers() {

  constexpr auto ret_type = constexpr_scan<0, std::tuple_size_v<Nodes>>(
    false,
    []<size_t I>(auto prefix) {
      using Node = std::tuple_element_t<I, Nodes>;
      using Type = Node::RetT;

      if constexpr (std::is_same_v<decltype(prefix), bool>) {
        return std::type_identity<std::tuple<Type>>{};
      } else {
        using Tp = decltype(prefix)::type;
        return std::type_identity<
          decltype(std::tuple_cat(Tp{}, std::tuple<Type>{}))>{};
      }
    }
  );

  return typename decltype(ret_type)::type{};
}


template<size_t nitems, size_t nthreads, class Nodes, class Ptrs>
void map(Ptrs& ptrs) {

  #pragma omp parallel for schedule(static) num_threads(nthreads)
  for (size_t tid = 0; tid < nthreads; ++tid) {

    size_t items_per_thread = (nitems + nthreads - 1) / nthreads;
    size_t start = std::min(items_per_thread * tid, nitems);
    size_t end = std::min(start + items_per_thread, nitems);

    auto buffers = create_buffers<Nodes>();

    for (; start < end; ++start) {
      constexpr_for<0, std::tuple_size_v<Nodes>>(
        [&ptrs, &buffers, start]<size_t I>() {
          using Node = std::tuple_element_t<I, Nodes>;
          typename Node::Op().template run<Node, Nodes>(ptrs, buffers, start);
        }
      );
    }
  }
}


template<class Nodes, class Op>
consteval auto filter_nodes() {

  auto ret = constexpr_scan<0, std::tuple_size_v<Nodes>>(
    false,
    []<size_t I>(auto prefix) {
      using Node = std::tuple_element_t<I, Nodes>;

      if constexpr (std::is_base_of_v<Op, typename Node::Op>) {

        if constexpr (std::is_same_v<decltype(prefix), bool>) {
          return std::type_identity<std::tuple<Node>>{};
        } else {
          using Tp = decltype(prefix)::type;
          return std::type_identity<
            decltype(std::tuple_cat(Tp{}, std::tuple<Node>{}))>{};
        }

      } else {
        return prefix;
      }
    }
  );

  static_assert(
    !std::is_same_v<decltype(ret), bool>, "No quilified nodes are found");

  return ret;
}


template<size_t nitems, size_t nthreads, class Nodes, class Ptrs>
void all_reduce(Ptrs& ptrs) {

  // create accumulator for each threads
  constexpr auto all_reduce_nodes = filter_nodes<Nodes, AllReduceOp>();
  using AllReduceNodes = decltype(all_reduce_nodes)::type;
  using AccBuffers = decltype(create_buffers<AllReduceNodes>());
  AccBuffers acc_buffers[nthreads];

  #pragma omp parallel for schedule(static) num_threads(nthreads)
  for (size_t tid = 0; tid < nthreads; ++tid) {

    size_t items_per_thread = (nitems + nthreads - 1) / nthreads;
    size_t start = std::min(items_per_thread * tid, nitems);
    size_t end = std::min(start + items_per_thread, nitems);

    auto buffers = create_buffers<Nodes>();

    for (; start < end; ++start) {
      constexpr_for<0, std::tuple_size_v<Nodes>>(
        [&ptrs, &buffers, start]<size_t I>() {
          using Node = std::tuple_element_t<I, Nodes>;
          typename Node::Op().template run<Node, Nodes>(ptrs, buffers, start);
        }
      );

      // accumulate in each thread
      constexpr_for<0, std::tuple_size_v<AllReduceNodes>>(
        [&buffers, &acc_buffers, tid]<size_t I>() {
          using Node = std::tuple_element_t<I, AllReduceNodes>;
          typename Node::Op().template
            acc<Node, Nodes, AllReduceNodes>(buffers, acc_buffers[tid]);
        }
      );
    }
  }

  // accumulate over threads
  constexpr_for<0, std::tuple_size_v<AllReduceNodes>>(
    [&ptrs, &acc_buffers]<size_t N>() {
      constexpr_for<0, nthreads>(
        [&ptrs, &acc_buffers](auto&& t) {
          using Node = std::tuple_element_t<N, AllReduceNodes>;
          typename Node::Op().template
            dump<Node, AllReduceNodes>(ptrs, acc_buffers[t.value]);
        }
      );
    }
  );
}


//---------------------------------------------------------------------
// Utility types
//---------------------------------------------------------------------

struct Coordinate
{
    int x;
    int y;
};


/**
 * Counting iterator
 */
template <
    typename ValueType,
    typename OffsetT = ptrdiff_t>
struct CountingInputIterator
{
    // Required iterator traits
    typedef CountingInputIterator               self_type;              ///< My own type
    typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

    ValueType val;

    /// Constructor
    inline CountingInputIterator(
        const ValueType &val)          ///< Starting value for the iterator instance to report
    :
        val(val)
    {}

    /// Postfix increment
    inline self_type operator++(int)
    {
        self_type retval = *this;
        val++;
        return retval;
    }

    /// Prefix increment
    inline self_type operator++()
    {
        val++;
        return *this;
    }

    /// Indirection
    inline reference operator*() const
    {
        return val;
    }

    /// Addition
    template <typename Distance>
    inline self_type operator+(Distance n) const
    {
        self_type retval(val + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    inline self_type& operator+=(Distance n)
    {
        val += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    inline self_type operator-(Distance n) const
    {
        self_type retval(val - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    inline self_type& operator-=(Distance n)
    {
        val -= n;
        return *this;
    }

    /// Distance
    inline difference_type operator-(self_type other) const
    {
        return val - other.val;
    }

    /// Array subscript
    template <typename Distance>
    inline reference operator[](Distance n) const
    {
        return val + n;
    }

    /// Structure dereference
    inline pointer operator->()
    {
        return &val;
    }

    /// Equal to
    inline bool operator==(const self_type& rhs)
    {
        return (val == rhs.val);
    }

    /// Not equal to
    inline bool operator!=(const self_type& rhs)
    {
        return (val != rhs.val);
    }

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        os << "[" << itr.val << "]";
        return os;
    }
};


template<class AIter, class BIter, class Offset, class Coord>
inline void merge_path_search(
  Offset diagonal, AIter a, BIter b, Offset a_len, Offset b_len, Coord& coord)
{
  Offset x_min = std::max(
    static_cast<int>(diagonal) - static_cast<int>(b_len), 0);
  Offset x_max = std::min(diagonal, a_len);

  while (x_min < x_max) {
    Offset x_pivot = (x_min + x_max) >> 1;
    if (a[x_pivot] <= b[diagonal - x_pivot - 1])
      x_min = x_pivot + 1;
    else
      x_max = x_pivot;
  }

  coord.x = std::min(x_min, a_len);
  coord.y = diagonal - x_min;
}


template<size_t nitems, size_t nrows, size_t nthreads, class Nodes, class Ptrs>
void reduce(Ptrs& ptrs) {

  constexpr auto reduce_nodes = filter_nodes<Nodes, ReduceOp>();
  using ReduceNodes = decltype(reduce_nodes)::type;
  static_assert(
    std::tuple_size_v<ReduceNodes> > 0, "No reducer in the pattern");

  // define value_carry_out and row_carry_out
  using ValueCarryOut = decltype(create_buffers<ReduceNodes>());
  ValueCarryOut value_carry_out[nthreads] = {};
  size_t row_carry_out[nthreads] = {};

  // row_end_offsets in all reducers are assumed to be the same
  using Node = std::tuple_element_t<0, ReduceNodes>;
  auto row_end_offsets =
    typename Node::Op().template get_row_end_offsets<Ptrs>(ptrs);

  #pragma omp parallel for schedule(static) num_threads(nthreads)
  for (size_t tid = 0; tid < nthreads; ++tid) {

    CountingInputIterator<size_t>  nonzero_indices(0);
    size_t num_merge_items = nrows + nitems;
    size_t items_per_thread = (num_merge_items + nthreads - 1) / nthreads;
    size_t start = std::min(items_per_thread * tid, num_merge_items);
    size_t end = std::min(start + items_per_thread, num_merge_items);

    Coordinate thread_coord;
    Coordinate thread_coord_end;
    merge_path_search(
      start, row_end_offsets, nonzero_indices, nrows, nitems, thread_coord);
    merge_path_search(
      end, row_end_offsets, nonzero_indices, nrows, nitems, thread_coord_end);

    for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x) {
      auto buffs = create_buffers<Nodes>();

      for (; thread_coord.y < row_end_offsets[thread_coord.x];
             ++thread_coord.y) {
        constexpr_for<0, std::tuple_size_v<Nodes>>(
          [&ptrs, &buffs, &thread_coord]<size_t I>() {
            using Node = std::tuple_element_t<I, Nodes>;
            typename Node::Op().template
              run<Node, Nodes>(ptrs, buffs, thread_coord.y);
          }
        );
      }

      // dump result to memory for row thread_coord.x
      constexpr_for<0, std::tuple_size_v<ReduceNodes>>(
        [&ptrs, &buffs, &thread_coord]<size_t I>() {
          using Node = std::tuple_element_t<I, ReduceNodes>;
          typename Node::Op().template
            dump<Node, Nodes>(ptrs, buffs, thread_coord.x);
        }
      );
    }

    // calculate and dump result to carry-out for the rest
    auto buffs = create_buffers<Nodes>();
    for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y) {
      constexpr_for<0, std::tuple_size_v<Nodes>>(
        [&ptrs, &buffs, &thread_coord]<size_t I>() {
          using Node = std::tuple_element_t<I, Nodes>;
          typename Node::Op().template
            run<Node, Nodes>(ptrs, buffs, thread_coord.y);
        }
      );

      row_carry_out[tid] = thread_coord_end.x;

      constexpr_for<0, std::tuple_size_v<ReduceNodes>>(
        [&value_carry_out, &buffs, &thread_coord_end, tid]<size_t I>() {
          using Node = std::tuple_element_t<I, ReduceNodes>;
          constexpr auto R = find<Node, Nodes>();
          std::get<I>(value_carry_out[tid]) = std::get<R>(buffs);
        }
      );
    }
  }

  // dump result to memory for row thread_coord.x
  for (size_t tid = 0; tid < nthreads - 1; ++tid) {
    if (row_carry_out[tid] < nrows) {
      constexpr_for<0, std::tuple_size_v<ReduceNodes>>(
        [&ptrs, &value_carry_out, &row_carry_out, tid]<size_t I>() {
          auto value = std::get<I>(value_carry_out[tid]);
          auto row = row_carry_out[tid];
          using Node = std::tuple_element_t<I, ReduceNodes>;
          typename Node::Op().template dump<Ptrs>(ptrs, value, row);
        }
      );
    }
  }
}

#endif