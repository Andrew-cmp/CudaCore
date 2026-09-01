CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
NCU ?= $(CUDA_HOME)/bin/ncu

GPU_ARCH ?= native
NVCCFLAGS ?= -O3 --use_fast_math -lineinfo -arch=$(GPU_ARCH)

KERNEL ?= gemm_test
M ?= 4096
N ?= 4096
K ?= 4096
WARMUP ?= 5
REPEAT ?= 1
NCU_SET ?= full

NATIVE_BUILD_DIR := build/native
REPORT_DIR := results/ncu
GEMM_BIN := $(NATIVE_BUILD_DIR)/gemm_ncu
CUBLAS_BIN := $(NATIVE_BUILD_DIR)/cublas_sgemm
REPORT_NAME := $(KERNEL)_$(M)x$(N)x$(K)
CUBLAS_REPORT_NAME := cublas_$(M)x$(N)x$(K)

GEMM_SOURCES := \
	ncu_runner.cu \
	gemm_test.cu \
	gemm_v0.cu \
	gemm_v1.cu \
	gemm_v2.cu \
	gemm_v3.cu \
	gemm_v4.cu \
	gemm_v5.cu \
	gemm_v2_warp_tiling.cu \
	gemm_v2_warp_tiling_swizzle_rw.cu \
	gemm_v2_warp_tiling_swizzle.cu

.PHONY: all gemm cublas run run-cublas ncu ncu-cublas clean

all: gemm cublas

gemm: $(GEMM_BIN)

cublas: $(CUBLAS_BIN)

$(GEMM_BIN): $(GEMM_SOURCES) | $(NATIVE_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(GEMM_SOURCES) -o $@

$(CUBLAS_BIN): cublas/sgemm_cublas.cu | $(NATIVE_BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< \
		-L$(CUDA_HOME)/lib64 -lcublas \
		-Xlinker -rpath -Xlinker $(CUDA_HOME)/lib64 \
		-o $@

$(NATIVE_BUILD_DIR):
	mkdir -p $@

$(REPORT_DIR):
	mkdir -p $@

run: $(GEMM_BIN)
	$(GEMM_BIN) $(KERNEL) $(M) $(N) $(K) $(WARMUP) $(REPEAT)

run-cublas: $(CUBLAS_BIN)
	$(CUBLAS_BIN) $(M) $(N) $(K) $(WARMUP) $(REPEAT)

ncu: $(GEMM_BIN) | $(REPORT_DIR)
	$(NCU) --set $(NCU_SET) \
		--target-processes all \
		--kernel-name-base function \
		--kernel-name $(KERNEL) \
		--launch-skip $(WARMUP) \
		--launch-count $(REPEAT) \
		--force-overwrite \
		--export $(REPORT_DIR)/$(REPORT_NAME) \
		$(GEMM_BIN) $(KERNEL) $(M) $(N) $(K) $(WARMUP) $(REPEAT)
	$(NCU) --import $(REPORT_DIR)/$(REPORT_NAME).ncu-rep \
		--page raw --csv > $(REPORT_DIR)/$(REPORT_NAME).csv

ncu-cublas: $(CUBLAS_BIN) | $(REPORT_DIR)
	$(NCU) --set $(NCU_SET) \
		--target-processes all \
		--launch-skip $(WARMUP) \
		--launch-count $(REPEAT) \
		--force-overwrite \
		--export $(REPORT_DIR)/$(CUBLAS_REPORT_NAME) \
		$(CUBLAS_BIN) $(M) $(N) $(K) $(WARMUP) $(REPEAT)
	$(NCU) --import $(REPORT_DIR)/$(CUBLAS_REPORT_NAME).ncu-rep \
		--page raw --csv > $(REPORT_DIR)/$(CUBLAS_REPORT_NAME).csv

clean:
	rm -rf $(NATIVE_BUILD_DIR) $(REPORT_DIR)
