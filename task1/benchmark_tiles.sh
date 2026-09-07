#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage:
  ./benchmark_tiles.sh H W K [TxXTy ...]

Examples:
  ./benchmark_tiles.sh 2048 2048 3
  ./benchmark_tiles.sh 1024 2048 5 64x8 64x16 128x16 256x16

Environment variables:
  CORENO=2          logical CPU used for every measurement
  RUNS=9            wall-clock repetitions used for each median
  PERF_RUNS=5       repetitions performed by perf stat -r
  KERNEL_ITERS=20   convolutions per perf process to amortize setup
  PERF=perf         perf command; use PERF='sudo perf' if required
EOF
}

if (( $# < 3 )); then
    usage
    exit 2
fi

H=$1
W=$2
K=$3
shift 3

CORENO=${CORENO:-2}
RUNS=${RUNS:-9}
PERF_RUNS=${PERF_RUNS:-5}
KERNEL_ITERS=${KERNEL_ITERS:-20}
PERF=${PERF:-perf}

if (( $# == 0 )); then
    TILE_PAIRS=(2048x2048 2048x128 2048x32 2048x16 2048x8 1024x2048 1024x512 1024x256 1024x128 1024x64 1024x32 1024x16)
else
    TILE_PAIRS=("$@")
fi

is_positive_integer()
{
    [[ $1 =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer()
{
    [[ $1 =~ ^(0|[1-9][0-9]*)$ ]]
}

if ! is_positive_integer "$H" ||
   ! is_positive_integer "$W" ||
   ! is_positive_integer "$K" ||
   ! is_nonnegative_integer "$CORENO" ||
   ! is_positive_integer "$RUNS" ||
   ! is_positive_integer "$PERF_RUNS" ||
   ! is_positive_integer "$KERNEL_ITERS"; then
    echo "ERROR: H, W, K, CORENO, RUNS, PERF_RUNS, and KERNEL_ITERS must be positive integers." >&2
    exit 2
fi

if (( K % 2 == 0 )); then
    echo "ERROR: K must be odd." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH="$SCRIPT_DIR/bin/benchmark_tile"

mkdir -p "$SCRIPT_DIR/bin"
g++ -std=c++17 -O2 -fno-tree-vectorize \
    -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw \
    -I"$SCRIPT_DIR/include" -Wall \
    "$SCRIPT_DIR/src/benchmark_tile.cpp" \
    "$SCRIPT_DIR/src/conv_naive.cpp" \
    "$SCRIPT_DIR/src/conv_tile.cpp" \
    -o "$BENCH"

read_field()
{
    local line=$1
    local key=$2
    awk -v key="$key" '{
        for (i = 1; i <= NF; ++i) {
            split($i, pair, "=")
            if (pair[1] == key) {
                print pair[2]
                exit
            }
        }
    }' <<<"$line"
}

perf_value()
{
    local file=$1
    local event=$2
    awk -F ';' -v wanted="$event" '
        $3 == wanted {
            value = $1
            gsub(/[[:space:],]/, "", value)
            split(value, parts, /[+-]/)
            print parts[1]
            exit
        }
    ' "$file"
}

run_perf()
{
    local implementation=$1
    local Tx=$2
    local Ty=$3
    local output_file=$4
    local -a perf_command
    read -r -a perf_command <<<"$PERF"

    if ! taskset -c "$CORENO" \
        "${perf_command[@]}" stat -r "$PERF_RUNS" -x ';' \
        -e instructions,L1-dcache-load-misses,branch-instructions,branch-misses -- \
        "$BENCH" kernel "$implementation" "$H" "$W" "$K" \
        "$Tx" "$Ty" "$KERNEL_ITERS" reserved \
        >/dev/null 2>"$output_file"; then
        echo "ERROR: perf failed. Check perf_event_paranoid or run with PERF='sudo perf'." >&2
        cat "$output_file" >&2
        exit 1
    fi
}

FIRST_PAIR=${TILE_PAIRS[0]}
if [[ ! $FIRST_PAIR =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
    echo "ERROR: Invalid tile pair '$FIRST_PAIR'; expected TxXTy, for example 128x16." >&2
    exit 2
fi

NAIVE_PERF=$(mktemp)
TILE_PERF=$(mktemp)
trap 'rm -f "$NAIVE_PERF" "$TILE_PERF"' EXIT

run_perf naive "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "$NAIVE_PERF"
NAIVE_INSTRUCTIONS=$(perf_value "$NAIVE_PERF" instructions)
NAIVE_L1_MISSES=$(perf_value "$NAIVE_PERF" L1-dcache-load-misses)
NAIVE_BRANCHES=$(perf_value "$NAIVE_PERF" branch-instructions)
NAIVE_BRANCH_MISSES=$(perf_value "$NAIVE_PERF" branch-misses)

if [[ -z $NAIVE_INSTRUCTIONS || -z $NAIVE_L1_MISSES ||
      -z $NAIVE_BRANCHES ||
      -z $NAIVE_BRANCH_MISSES ]]; then
    echo "ERROR: perf did not return all requested counters." >&2
    cat "$NAIVE_PERF" >&2
    exit 1
fi

NAIVE_MPKI=$(awk -v misses="$NAIVE_L1_MISSES" -v instructions="$NAIVE_INSTRUCTIONS" \
    'BEGIN { printf "%.3f", 1000.0 * misses / instructions }')
NAIVE_BRANCH_MPKI=$(awk -v misses="$NAIVE_BRANCH_MISSES" -v instructions="$NAIVE_INSTRUCTIONS" \
    'BEGIN { printf "%.3f", 1000.0 * misses / instructions }')

echo "Workload: H=$H W=$W K=$K  CPU=$CORENO"
echo "Timing repetitions=$RUNS  perf repetitions=$PERF_RUNS  kernels/perf-run=$KERNEL_ITERS"
printf '%6s %6s %8s %11s %11s %9s %12s %12s %12s %12s %15s %15s %15s %15s\n' \
    "Tx" "Ty" "naive(ms)" "tile(ms)" "speedup" \
    "naive L1MPKI" "tile L1MPKI" "naive BrMPKI" "tile BrMPKI" \
    "naive branches" "tile branches" "naive br-miss" "tile br-miss"

for pair in "${TILE_PAIRS[@]}"; do
    if [[ ! $pair =~ ^([1-9][0-9]*)x([1-9][0-9]*)$ ]]; then
        echo "ERROR: Invalid tile pair '$pair'; expected TxXTy, for example 128x16." >&2
        exit 2
    fi
    Tx=${BASH_REMATCH[1]}
    Ty=${BASH_REMATCH[2]}

    timing=$(taskset -c "$CORENO" \
        "$BENCH" time "$H" "$W" "$K" "$Tx" "$Ty" "$RUNS" reserved)
    correct=$(read_field "$timing" correct)
    naive_ms=$(read_field "$timing" naive_ms)
    tile_ms=$(read_field "$timing" tile_ms)
    speedup=$(read_field "$timing" speedup)

    if [[ $correct != yes ]]; then
        printf '%6d %6d %8s\n' "$Tx" "$Ty" "NO"
        continue
    fi

    run_perf tile "$Tx" "$Ty" "$TILE_PERF"
    tile_instructions=$(perf_value "$TILE_PERF" instructions)
    tile_l1_misses=$(perf_value "$TILE_PERF" L1-dcache-load-misses)
    tile_branches=$(perf_value "$TILE_PERF" branch-instructions)
    tile_branch_misses=$(perf_value "$TILE_PERF" branch-misses)

    if [[ -z $tile_instructions || -z $tile_l1_misses ||
          -z $tile_branches ||
          -z $tile_branch_misses ]]; then
        echo "ERROR: perf did not return all requested counters for ${Tx}x${Ty}." >&2
        cat "$TILE_PERF" >&2
        exit 1
    fi

    tile_mpki=$(awk -v misses="$tile_l1_misses" -v instructions="$tile_instructions" \
        'BEGIN { printf "%.3f", 1000.0 * misses / instructions }')
    tile_branch_mpki=$(awk -v misses="$tile_branch_misses" -v instructions="$tile_instructions" \
        'BEGIN { printf "%.3f", 1000.0 * misses / instructions }')

    printf '%6d %6d %11.3f %11.3f %9.3f %12s %12s %12s %12s %15.0f %15.0f %15.0f %15.0f\n' \
        "$Tx" "$Ty" "$naive_ms" "$tile_ms" "$speedup" \
        "$NAIVE_MPKI" "$tile_mpki" "$NAIVE_BRANCH_MPKI" "$tile_branch_mpki" \
        "$NAIVE_INSTRUCTIONS" "$tile_instructions" \
        "$NAIVE_BRANCH_MISSES" "$tile_branch_misses"
done
