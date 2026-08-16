NVCC = nvcc
CXXFLAGS = -std=c++14 -O3
NVCCFLAGS = -arch=sm_70 --ptxas-options=-v

# 目标文件
TARGETS = prefill_mha verify_mha mha_gqa

all: $(TARGETS)

prefill_mha: prefill_mha.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $<

verify_mha: verify_mha.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $<

mha_gqa: mha_gqa_corrected.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean 