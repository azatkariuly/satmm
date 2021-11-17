#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
__global__ void gpu_matrix_mult(const float *__restrict__ A,
                        const float *__restrict__ X,
                        float *__restrict__ O, 
                        int *__restrict__ maskA, int *__restrict__ maskX, 
                        const int m, const int n, const int k,
                        const int t, const float minClip, const float maxClip)
{
  int isSat = 0;
  int satLoc = -1;
  int numTile = (n + t - 1) / t;
  int cbatch = blockIdx.x * m * k;
  int abatch = blockIdx.x * m * n;
  int row = blockIdx.z * blockDim.y + threadIdx.y; 
  int col = blockIdx.y * blockDim.x + threadIdx.x;
  float sum = 0; 

  if(col < k && row < m) 
  {
    for(int i = 0; i < n; i+=t) 
    {
      float psum = 0;
      for(int j = i; j < i+t && j < n; j++)
      {
        psum += A[abatch + row * n + j] * X[j * k + col];
      }
      sum = max(minClip, min(maxClip, sum + psum)); 
      isSat = (minClip == sum) || (sum == maxClip);
      satLoc = i * isSat + (!i) * isSat;
    }
    O[cbatch + row * k + col] = sum;
    for(int i = satLoc + 1; i < numTile; i++)
    {
      for(int j = i * t; j < (i+1) * t && j < n; j++)
      {
        atomicAdd(maskA + abatch + row * n + j, 1);
        atomicAdd(maskX + j * k + col, 1);
      }
    }  
  }
}


std::vector<torch::Tensor> satmm_cuda_forward(
                                              torch::Tensor A,
                                              torch::Tensor X,
                                              int t, int bits)
{
  const int b = A.size(0);
  const int m = A.size(1);
  const int n = A.size(2);
  const int k = X.size(1);

  torch::Tensor O = torch::empty({b, m, k}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  torch::Tensor maskX = torch::zeros({n, k}, torch::device(torch::kCUDA).dtype(torch::kInt32));
  torch::Tensor maskA = torch::zeros({b, m, n}, torch::device(torch::kCUDA).dtype(torch::kInt32));

  dim3 dimGrid(b, (k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE); //batch, col, row
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  gpu_matrix_mult<<<dimGrid, dimBlock>>>(
                                          A.data_ptr<float>(), 
                                          X.data_ptr<float>(),
                                          O.data_ptr<float>(), 
                                          maskA.data_ptr<int>(), maskX.data_ptr<int>(),
                                          m, n, k,
                                          t, (float) 0-(1<<(bits-1)), (float) (1<<(bits-1))-1);
  return {O, maskA, maskX};
}
//import torch;import satmm_cuda;A = torch.rand((2,8,16),dtype=torch.float32).cuda()-0.5;B = torch.rand((16,6),dtype=torch.float32).cuda()-0.5;O,maskA,maskX=satmm_cuda.forward(A,B,2,1);maskA,maskX
//import torch;import satmm_cuda;A = torch.rand((128,1024,4096),dtype=torch.float32).cuda()-0.5;B = torch.rand((4096,2048),dtype=torch.float32).cuda()-0.5;O,maskA,maskX=satmm_cuda.forward(A,B,128,8);maskA,maskX
