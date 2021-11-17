#include <torch/extension.h>

#include <iostream>
#include <vector>


// A * X, b m n * n k -> b m k  
std::vector<torch::Tensor> satmm_cuda_forward(
    torch::Tensor A,
    torch::Tensor X,
    int t, int bits);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_DIM(n, t) AT_ASSERTM(n <= 48 * t, "matrix size should be smaller than 48 * (tile_size)")
#define CHECK_INPUT(x) CHECK_CUDA(x);// CHECK_CONTIGUOUS(x)
//std::vector<torch::Tensor>
std::vector<torch::Tensor> satmm_forward(
    torch::Tensor A,
    torch::Tensor X,
    int t, int bits) {
  CHECK_INPUT(A);
  CHECK_INPUT(X);
  //return AX, maskA, maskX
  return satmm_cuda_forward(A, X, t, bits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &satmm_forward, "MM with SA");
}
