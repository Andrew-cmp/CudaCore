 #include <cuda_runtime.h>
 #include <iostream>
 #include <vector>
 #include <algorithm>
 //http://www.zh0ngtian.tech/posts/51050901.html
 #ifndef BLOCK_SIZE
 #define BLOCK_SIZE 1024
 #endif
 
 // 来自 SanThenFan.cu
 __global__ void ScanPart(int *input, int *part, int *output, int n, int part_num);
 
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
 
   // 启动 Kernel（仅做分段内扫描与段和输出）
   ScanPart<<<grid_size, block_size>>>(d_in, d_part, d_out, n, part_num);
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


