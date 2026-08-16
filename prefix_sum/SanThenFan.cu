#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif
__device__ int ScanWarp(int val) {
    int lane_id = threadIdx.x % 32;
    int tmp = __shfl_up_sync(0xffffffff, val, 1);
    if (lane_id >= 1) {
      val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 2);
    if (lane_id >= 2) {
      val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 4);
    if (lane_id >= 4) {
      val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 8);
    if (lane_id >= 8) {
      val += tmp;
    }
    tmp = __shfl_up_sync(0xffffffff, val, 16);
    if (lane_id >= 16) {
      val += tmp;
    }
    return val;
  }
  
  __device__ __forceinline__ int ScanBlock(int val) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    extern __shared__ int warp_sum[];
  
    val = ScanWarp(val);
    __syncthreads();
  
    if (lane_id == 31) {
      warp_sum[warp_id] = val;
    }
    __syncthreads();
  
    if (warp_id == 0) {
      warp_sum[lane_id] = ScanWarp(warp_sum[lane_id]);
    }
    __syncthreads();
  
    if (warp_id > 0) {
      val += warp_sum[warp_id - 1];
    }
    __syncthreads();
    return val;
  }
  
  __global__ void ScanPart(int *input, int *part, int *output, int n, int part_num) {
    for (int part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
      int tid = part_i * blockDim.x + threadIdx.x;
      int val = tid < n ? input[tid] : 0;
      val = ScanBlock(val);
      __syncthreads();
      if (tid < n) {
        output[tid] = val;
      }
      if (threadIdx.x == blockDim.x - 1) {
        part[part_i] = val;
      }
    }
  }
  __global__ void AddPartSum(int *part, int *output, int n, int part_num) {
    for (int part_i = blockIdx.x; part_i < part_num; part_i += gridDim.x) {
      if (part_i == 0) {
        continue;
      }
      int tid = part_i * blockDim.x + threadIdx.x;
      if (tid < n) {
        output[tid] += part[part_i - 1];
      }
    }
  }
  void PrefixSum(int *input, int *part, int *output, int n) {
    int part_num = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_num = std::min<int>(part_num, 128);
    int shm_size = 32 * sizeof(int); // 32 个 warp 和

    // 第 1 步：分段包含式扫描，得到 output 与各段总和 part
    ScanPart<<<block_num, BLOCK_SIZE, shm_size>>>(input, part, output, n, part_num);
    cudaDeviceSynchronize();

    // 第 2 步：对 part 做包含式前缀和（在 Host 上完成，避免递归额外缓冲区管理）
    if (part_num >= 2) {
      std::vector<int> host_part(part_num);
      cudaMemcpy(host_part.data(), part, sizeof(int) * part_num, cudaMemcpyDeviceToHost);
      for (int i = 1; i < part_num; ++i) host_part[i] += host_part[i - 1];
      cudaMemcpy(part, host_part.data(), sizeof(int) * part_num, cudaMemcpyHostToDevice);

      // 第 3 步：将段偏移加回（第 0 段不加）
      AddPartSum<<<block_num, BLOCK_SIZE>>>(part, output, n, part_num);
      cudaDeviceSynchronize();
    }
  }

#define CUDA_CHECK(call) \
 do { \
   cudaError_t _e = (call); \
   if (_e != cudaSuccess) { \
     std::cerr << "CUDA error " << cudaGetErrorName(_e) << ": " \
               << cudaGetErrorString(_e) \
               << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
     std::exit(1); \
   } \
 } while (0)
int main() {
    const int n = 1 << 20; // 1048576
    const int block_size = BLOCK_SIZE;
    const int part_num = (n + block_size - 1) / block_size;
    const int grid_size = std::max(1, std::min(part_num, 128));
    // 准备输入
    std::vector<int> h_in(n);
    for (int i = 0; i < n; ++i) h_in[i] = 1; // 全 1，便于观察
    std::vector<int> h_out(n, 0);
    std::vector<int> h_part(part_num, 0);
    // 设备内存
    int *d_in = nullptr, *d_out = nullptr, *d_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(int) * std::max(1, n)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(int) * std::max(1, n)));
    CUDA_CHECK(cudaMalloc(&d_part, sizeof(int) * std::max(1, part_num)));
    if (n > 0) CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(int) * std::max(1, n)));
    CUDA_CHECK(cudaMemset(d_part, 0, sizeof(int) * std::max(1, part_num)));
  // 计算全局包含式前缀和
    PrefixSum(d_in, d_part, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // 回读结果
    if (n > 0) CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(int) * n, cudaMemcpyDeviceToHost));
    if (part_num > 0) CUDA_CHECK(cudaMemcpy(h_part.data(), d_part, sizeof(int) * part_num, cudaMemcpyDeviceToHost));
    // 打印部分信息
    std::cout << "n=" << n << ", BLOCK_SIZE=" << block_size
              << ", grid_size=" << grid_size << ", part_num=" << part_num << "\n";
    if (n > 0) {
      std::cout << "out[0]=" << h_out[0]
                << ", out[BLOCK_SIZE-1]=" << h_out[block_size - 1]
                << ", out[BLOCK_SIZE]=" << h_out[block_size] << "\n";
    }
    if (part_num > 0) {
      std::cout << "part[0]=" << h_part[0];
      if (part_num > 1) std::cout << ", part[1]=" << h_part[1];
      std::cout << "\n";
    }
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_part));
    return 0;

}