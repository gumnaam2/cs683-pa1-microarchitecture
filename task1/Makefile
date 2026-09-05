# CS683 PA-1 Task 1  build file.
#
# The compiler flags are PINNED and identical for every stage so that all speedups
# are attributable to the techniques you implement, not to compiler options:
#
#   -O2                 realistic scalar optimization (inlining, register allocation)
#   -fno-tree-vectorize compiler auto-vectorization is OFF, so the SIMD stage is YOUR job
#   -mavx2 -mfma        enable the AVX2 + FMA intrinsics your conv_simd/optimized use
#
# Do NOT add -march=native or -ftree-vectorize: the autograder uses exactly these flags.

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw -Iinclude -Wall
# CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall
LDFLAGS  :=

BIN          := bin
MAIN         := src/main.cpp
STUDENT_SRC  := src/conv_naive.cpp src/conv_reorder.cpp src/conv_unroll.cpp \
                src/conv_tile.cpp src/conv_simd.cpp src/conv_optimized.cpp

# Instructor-only: reference implementations used to calibrate/regress the tiers.
SOLUTION_SRC := src/conv_naive.cpp solution/conv_reorder.cpp solution/conv_unroll.cpp \
                solution/conv_tile.cpp solution/conv_simd.cpp solution/conv_optimized.cpp

NUM_INSTR_SRC := src/num_instr.cpp src/conv_naive.cpp src/conv_simd.cpp

.PHONY: all run solution instr clean

all: $(BIN)/conv $(BIN)/num_instr

# Student build: edit src/conv_*.cpp, then `make` and `./bin/conv`.
$(BIN)/conv: $(STUDENT_SRC) $(MAIN) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN)/num_instr: $(NUM_INSTR_SRC) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

instr: $(BIN)/num_instr

run: $(BIN)/conv
	./$(BIN)/conv

# Instructor build against solution/ (kept out of the student handout).
solution: $(BIN)/conv_solution
$(BIN)/conv_solution: $(SOLUTION_SRC) $(MAIN) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
