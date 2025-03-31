#include "../include/cuda_backend.cuh"

#define EASIER_FUNC(name, ParamT, param) \
  void name(ParamT& param) { \
    eaiser_cuda_func<ParamT>(param); \
  }

template<typename ParamT>
__forceinline__ void eaiser_cuda_func(ParamT& param) {
  typedef typename ParamT::ValueT ValueT;
  typedef typename ParamT::OffsetT OffsetT;

  typedef cub::ReduceByKeyScanTileState<ValueT, int> ScanTileStateT;
  typedef cub::KeyValuePair<int, ValueT> KeyValuePairT;
  typedef typename cub::CubVector<int, 2>::Type CoordinateT;
  // typedef cub::SpmvParams<ValueT, int> SpmvParamsT;
  typedef cub::DispatchSpmv<ValueT, int> DispatchSpmvT;

  // ParamT must have num_rows and num_nonzeros
  auto config = get_config<ValueT>(param.num_rows, param.num_nonzeros);
  auto num_merge_tiles = std::get<0>(config);
  auto num_segment_fixup_tiles = std::get<1>(config);
  auto spmv_grid_size = std::get<2>(config);
  auto segment_fixup_grid_size = std::get<3>(config);
  auto spmv_config = std::get<4>(config);
  auto segment_fixup_config = std::get<5>(config);

  auto mem = allocate_reducer_memory<ValueT>(num_segment_fixup_tiles, num_merge_tiles);
  auto d_temp_storage = std::get<0>(mem);
  auto d_tile_carry_pairs = std::get<1>(mem);
  auto tile_state = std::get<2>(mem);
  auto d_tile_coordinates = allocate_tile_coordinates<CoordinateT>(num_merge_tiles);

  // ParamT must have d_row_end_offsets
  int search_grid_size = cub::DivideAndRoundUp(num_merge_tiles + 1, 128);
  THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
    search_grid_size, 128, 0, cudaStreamDefault
  ).doit(
    cub::DeviceSpmvSearchKernel<
      DispatchSpmvT::PtxSpmvPolicyT, int, CoordinateT, ParamT>,
    num_merge_tiles,
    d_tile_coordinates,
    param);

  // Check for failure to launch
  CubDebug(cudaPeekAtLastError());

  // Invoke spmv_kernel
  // THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
  //     spmv_grid_size, spmv_config.block_threads, 0, cudaStreamDefault
  // ).doit(
  //   kernel<DispatchSpmvT::PtxSpmvPolicyT, ScanTileStateT, ValueT, int>,
  //   // cub::DeviceSpmvKernel<DispatchSpmvT::PtxSpmvPolicyT, ScanTileStateT,
  //   //   ValueT, int, CoordinateT, false, false>,
  //   spmv_params,
  //   d_tile_coordinates,
  //   d_tile_carry_pairs,
  //   num_merge_tiles,
  //   tile_state,
  //   num_segment_fixup_tiles);

  // Check for failure to launch
  CubDebug(cudaPeekAtLastError());

  CubDebug(cudaFree(d_temp_storage));
  CubDebug(cudaFree(d_tile_coordinates));
}
