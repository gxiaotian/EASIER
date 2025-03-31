#include <torch/extension.h>
#include <mkl.h>

void spmv(
    int rows,
    int cols,
    int nnz,
    torch::Tensor& values,
    torch::Tensor& col_idx,
    torch::Tensor& row_ptr,
    torch::Tensor& vec,
    torch::Tensor& ret
) {
  sparse_matrix_t A_mkl = NULL;
  sparse_status_t res = mkl_sparse_d_create_csr(
    &A_mkl, SPARSE_INDEX_BASE_ZERO, rows, cols,
    row_ptr.data_ptr<int>(), row_ptr.data_ptr<int>() + 1,
    col_idx.data_ptr<int>(), values.data_ptr<double>());

  matrix_descr dsc;
  dsc.type = SPARSE_MATRIX_TYPE_GENERAL;
  sparse_status_t stat = mkl_sparse_d_mv(
    SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_mkl, dsc,
    vec.data_ptr<double>(), 0.0, ret.data_ptr<double>());

  mkl_sparse_destroy(A_mkl);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmv", &spmv);
}