CC ?= gcc
HIPCC ?= /opt/rocm/bin/hipcc
ROCM_ARCH ?= gfx1100
# /etc/profile.d/rocm.sh sets this in login shells; non-login shells need it.
export ROCM_PATH ?= /opt/rocm
CFLAGS ?= -O2 -g -Wall -Wextra -Wshadow -Wno-unused-parameter -pthread -fopenmp
CFLAGS += -I src
HIPFLAGS ?= -O3 -ffast-math -fno-finite-math-only -std=c++17 --offload-arch=$(ROCM_ARCH)
HIPFLAGS += -D__HIP_PLATFORM_AMD__ -DQ4_HIP -I src -Wno-unused-parameter
LDFLAGS ?= -pthread -fopenmp -lm

SRC = src/q4_gguf.c src/q4_place.c src/q4_expert.c src/q4_ple.c src/q4_quant.c src/q4_moe.c src/q4_store.c src/q4_fwd.c src/q4_tok.c src/q4_engine.c src/q4_mtp.c src/q4_server.c src/q4_vision.c src/q4_crash.c
OBJ = $(SRC:.c=.o)
HIP_OBJ = src/q4_rocm.o src/q4_xg.o src/q4_mmvq.o
CPUEXP_OBJ = src/q4_cpuexp.o

# MMQ kernels (MIT), vendored under third_party/llama.cpp. See README.
# q4_mmq.cu wrapper + quantize.cu + one object per covered expert type.
MMQ_GG = third_party/llama.cpp/ggml
MMQ_CU = $(MMQ_GG)/src/ggml-cuda
MMQ_INC = -I $(MMQ_GG)/include -I $(MMQ_GG)/src -I $(MMQ_CU)
MMQ_TYPES = q2_0 iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0
MMQ_OBJ = src/q4_mmq.o $(MMQ_CU)/quantize.o \
    $(MMQ_TYPES:%=$(MMQ_CU)/template-instances/mmq-instance-%.o)

# hipBLASLt/hipBLAS dense batched GEMM for prefill.
HGEMM_OBJ = src/q4_gemm.o
HGEMM_LIB = -lhipblaslt -lhipblas

.PHONY: all test clean inspect plan

all: q4 q4-test

q4: $(OBJ) src/q4_cli.o $(HIP_OBJ) $(CPUEXP_OBJ) $(MMQ_OBJ) $(HGEMM_OBJ)
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(LDFLAGS) $(HGEMM_LIB)

q4-test: $(OBJ) src/q4_test.o $(HIP_OBJ) $(CPUEXP_OBJ) $(MMQ_OBJ) $(HGEMM_OBJ)
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(LDFLAGS) $(HGEMM_LIB)

src/q4_cpuexp.o: src/q4_cpuexp.c src/q4.h
	$(CC) $(CFLAGS) -O3 -mavx2 -mfma -mf16c -c -o $@ $<

src/q4_tok.o: src/q4_ucat.inc

src/%.o: src/%.c src/q4.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/q4_rocm.o: src/q4_rocm.cu src/q4.h
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

src/q4_xg.o: src/q4_xg.cu src/q4.h src/q4_iq_grids.inc
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

src/q4_mmq.o: src/q4_mmq.cu src/q4.h
	$(HIPCC) $(HIPFLAGS) -DGGML_USE_HIP=1 $(MMQ_INC) -c -o $@ $<

src/q4_mmvq.o: src/q4_mmvq.cu src/q4.h
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

src/q4_gemm.o: src/q4_gemm.cu src/q4.h src/q4_hgemm_tuning.hpp
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

$(MMQ_CU)/quantize.o: $(MMQ_CU)/quantize.cu
	$(HIPCC) $(HIPFLAGS) -DGGML_USE_HIP=1 $(MMQ_INC) -c -o $@ $<

$(MMQ_CU)/template-instances/mmq-instance-%.o: $(MMQ_CU)/template-instances/mmq-instance-%.cu
	$(HIPCC) $(HIPFLAGS) -DGGML_USE_HIP=1 $(MMQ_INC) -c -o $@ $<

# The model download_q4.sh fetches (GSQ-RCO IQ3_S, or IQ3E with `iq3e`).
MODEL_GSQ := GSQ-RCO/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
MODEL_IQ3E := IQ3E-Q8D-MTP/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
MODEL ?= $(firstword $(wildcard models/$(MODEL_GSQ) ../models/$(MODEL_GSQ) models/$(MODEL_IQ3E) ../models/$(MODEL_IQ3E)) models/$(MODEL_GSQ))

test: q4-test
	./q4-test
	./q4-test "$(MODEL)"

inspect: q4
	./q4 inspect "$(MODEL)"

plan: q4
	./q4 plan "$(MODEL)" --ctx 8192

clean:
	rm -f q4 q4-test src/*.o $(MMQ_OBJ)
