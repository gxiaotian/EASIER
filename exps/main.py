import subprocess
import time
from tqdm import trange

import numpy as np
import scipy.sparse

import torch
from torch.utils.cpp_extension import load


# load matrix
# coo_mat = scipy.io.mmread('HV15R.mtx')
# scipy.sparse.save_npz('hv15r.npz', coo_mat)
coo_mat = scipy.sparse.load_npz('./hv15r.npz')
# coo_mat = scipy.sparse.load_npz("./rm07r.npz")

# coo_mat = scipy.sparse.coo_matrix(
#     (np.linspace(0, 1, 11),
#      ([0, 1, 2, 3, 4, 1, 2, 3, 2, 3, 1],
#       [0, 1, 2, 3, 4, 2, 1, 2, 3, 1, 3])),
#     shape=(5, 5)
# )
csr_mat = coo_mat.tocsr()

print(csr_mat.shape[0])
print(csr_mat.nnz)

vec = np.ones(csr_mat.shape[0])

# scipy baseline
scipy_ret = torch.from_numpy(csr_mat.dot(vec))

# mkl
mkl_ext = load(
    name="mkl_extension",
    verbose=False,
    sources=['mkl_binding.cpp'],
    extra_cflags=['-O3', '-std=c++20', '-I/usr/include/mkl'],
)

mkl_ret = torch.zeros(csr_mat.shape[0]).double()
for i in range(10):
    mkl_ext.spmv(
        csr_mat.shape[0],
        csr_mat.shape[0],
        csr_mat.nnz,
        torch.from_numpy(csr_mat.data).double(),
        torch.from_numpy(csr_mat.indices),
        torch.from_numpy(csr_mat.indptr),
        torch.from_numpy(vec).double(),
        mkl_ret)

start = time.time()
for i in trange(1000):
    mkl_ext.spmv(
        csr_mat.shape[0],
        csr_mat.shape[0],
        csr_mat.nnz,
        torch.from_numpy(csr_mat.data).double(),
        torch.from_numpy(csr_mat.indices),
        torch.from_numpy(csr_mat.indptr),
        torch.from_numpy(vec).double(),
        mkl_ret)
end = time.time()

print('mkl time:\t', (end - start) / 1000)
print('mkl error:\t', (scipy_ret - mkl_ret).norm())

# easier
try:
    subprocess.run(
        ['icpx', '-shared', '-march=native', '-mavx', '-fiopenmp',  # icpx -fiopenmp
         '-O3', '-fPIC', '-D_GLIBCXX_USE_CXX11_ABI=0', '-std=c++20',
         '-o', 'kernel.so', 'kernel.cpp',],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    raise Exception(e.stderr.decode())


esr_ext = load(
    name="easier_cpu_extension",
    verbose=False,
    sources=['binding.cpp'],
    extra_cflags=['-O3', '-std=c++20'],
    extra_ldflags=['/root/data/myEASIER/exps/kernel.so']
)

esr_ret = torch.zeros(csr_mat.shape[0]).double()
for i in range(10):
    esr_ext.foo((
        torch.from_numpy(vec).double(),
        torch.from_numpy(csr_mat.indices),
        torch.from_numpy(csr_mat.data).double(),
        torch.from_numpy(csr_mat.indptr)[1:],
        esr_ret))

start = time.time()
for i in trange(1000):
    esr_ext.foo((
        torch.from_numpy(vec).double(),
        torch.from_numpy(csr_mat.indices),
        torch.from_numpy(csr_mat.data).double(),
        torch.from_numpy(csr_mat.indptr)[1:],
        esr_ret))
end = time.time()

print('esr time:\t', (end - start) / 1000)
print('esr error;\t', (scipy_ret - esr_ret).norm())
