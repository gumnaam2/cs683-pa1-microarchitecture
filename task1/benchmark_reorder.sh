#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage:
  ./benchmark_reorder.sh [H W K]

Defaults: H=2048 W=2048 K=3

Environment variables:
  CORENO=2          logical CPU used for every measurement
  RUNS=9            wall-clock repetitions
  PERF_RUNS=5       perf stat repetitions
  KERNEL_ITERS=50   convolutions executed by each perf process
  PERF=perf         perf command; use PERF='sudo perf' if required
EOF
}

if (( $# != 0 && $# != 3 )); then
    usage
    exit 2
fi

H=${1:-2048}
W=${2:-2048}
K=${3:-3}
CORENO=${CORENO:-2}
RUNS=${RUNS:-9}
PERF_RUNS=${PERF_RUNS:-5}
KERNEL_ITERS=${KERNEL_ITERS:-50}
PERF=${PERF:-perf}
export LC_ALL=C

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
    echo "ERROR: dimensions and repetition counts must be positive integers; CORENO may be zero." >&2
    exit 2
fi

if (( K % 2 == 0 )); then
    echo "ERROR: K must be odd." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BENCH="$SCRIPT_DIR/bin/benchmark_reorder"
mkdir -p "$SCRIPT_DIR/bin"

g++ -std=c++17 -O2 -fno-tree-vectorize \
    -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw \
    -I"$SCRIPT_DIR/include" -Wall \
    "$SCRIPT_DIR/src/benchmark_reorder.cpp" \
    "$SCRIPT_DIR/src/conv_naive.cpp" \
    "$SCRIPT_DIR/src/conv_reorder.cpp" \
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

event_value()
{
    local file=$1
    local wanted=$2
    awk -F ';' -v wanted="$wanted" '
        {
            event = $3
            gsub(/[[:space:]]/, "", event)
            sub(/:u$/, "", event)
            if (event == wanted) {
                value = $1
                gsub(/[[:space:],]/, "", value)
                split(value, pieces, /[+-]/)
                if (pieces[1] !~ /^</)
                    print pieces[1]
                exit
            }
        }
    ' "$file"
}

divide()
{
    local numerator=$1
    local denominator=$2
    local decimals=${3:-3}
    if [[ -z $numerator || -z $denominator ]]; then
        printf 'N/A'
    else
        awk -v n="$numerator" -v d="$denominator" -v p="$decimals" \
            'BEGIN { printf "%.*f", p, n / d }'
    fi
}

multiply_divide()
{
    local first=$1
    local multiplier=$2
    local denominator=$3
    local decimals=${4:-3}
    if [[ -z $first || -z $denominator ]]; then
        printf 'N/A'
    else
        awk -v a="$first" -v m="$multiplier" -v d="$denominator" -v p="$decimals" \
            'BEGIN { printf "%.*f", p, a * m / d }'
    fi
}

run_perf_group()
{
    local implementation=$1
    local events=$2
    local output_file=$3
    local required=$4
    local temporary
    temporary=$(mktemp)
    local -a perf_command
    read -r -a perf_command <<<"$PERF"

    if taskset -c "$CORENO" \
        "${perf_command[@]}" stat -r "$PERF_RUNS" -x ';' \
        -e "$events" -- \
        "$BENCH" kernel "$implementation" "$H" "$W" "$K" \
        "$KERNEL_ITERS" reserved \
        >/dev/null 2>"$temporary"; then
        cat "$temporary" >>"$output_file"
    elif [[ $required == yes ]]; then
        echo "ERROR: perf failed for required events: $events" >&2
        cat "$temporary" >&2
        rm -f "$temporary"
        exit 1
    else
        echo "WARNING: optional perf events unavailable: $events" >&2
    fi
    rm -f "$temporary"
}

collect_perf()
{
    local implementation=$1
    local output_file=$2
    : >"$output_file"

    # Keep each group within the usual number of hardware counters so perf does
    # not multiplex events and distort comparisons between implementations.
    run_perf_group "$implementation" \
        'instructions:u,cycles:u,branch-instructions:u,branch-misses:u' \
        "$output_file" yes
    run_perf_group "$implementation" \
        'L1-dcache-loads:u,L1-dcache-load-misses:u,cache-references:u,cache-misses:u' \
        "$output_file" yes
    run_perf_group "$implementation" \
        'stalled-cycles-frontend:u,stalled-cycles-backend:u' \
        "$output_file" no
    run_perf_group "$implementation" \
        'ls_dispatch.store_dispatch:u' \
        "$output_file" no
}

TIMING=$(taskset -c "$CORENO" \
    "$BENCH" time "$H" "$W" "$K" "$RUNS" reserved)

if [[ $(read_field "$TIMING" correct) != yes ]]; then
    echo "ERROR: conv_reorder does not match conv_naive." >&2
    exit 1
fi

NAIVE_PERF=$(mktemp)
REORDER_PERF=$(mktemp)
trap 'rm -f "$NAIVE_PERF" "$REORDER_PERF"' EXIT

collect_perf naive "$NAIVE_PERF"
collect_perf reorder "$REORDER_PERF"

declare -A N R
EVENTS=(instructions cycles branch-instructions branch-misses \
        L1-dcache-loads L1-dcache-load-misses cache-references cache-misses \
        stalled-cycles-frontend stalled-cycles-backend ls_dispatch.store_dispatch)

for event in "${EVENTS[@]}"; do
    N[$event]=$(event_value "$NAIVE_PERF" "$event")
    R[$event]=$(event_value "$REORDER_PERF" "$event")
done

for required_event in instructions cycles branch-instructions branch-misses \
                      L1-dcache-loads L1-dcache-load-misses; do
    if [[ -z ${N[$required_event]} || -z ${R[$required_event]} ]]; then
        echo "ERROR: perf did not return required event '$required_event'." >&2
        exit 1
    fi
done

naive_mean=$(read_field "$TIMING" naive_mean_ms)
reorder_mean=$(read_field "$TIMING" reorder_mean_ms)
naive_median=$(read_field "$TIMING" naive_median_ms)
reorder_median=$(read_field "$TIMING" reorder_median_ms)
naive_cv=$(read_field "$TIMING" naive_cv)
reorder_cv=$(read_field "$TIMING" reorder_cv)
speedup=$(read_field "$TIMING" speedup)
flops=$(awk -v h="$H" -v w="$W" -v k="$K" 'BEGIN { print 2.0 * h * w * k * k }')
naive_gflops=$(multiply_divide "$flops" 1 "$naive_mean" 3)
naive_gflops=$(multiply_divide "$naive_gflops" 0.000001 1 3)
reorder_gflops=$(multiply_divide "$flops" 1 "$reorder_mean" 3)
reorder_gflops=$(multiply_divide "$reorder_gflops" 0.000001 1 3)

printf 'Workload: H=%d W=%d K=%d  CPU=%d\n' "$H" "$W" "$K" "$CORENO"
printf 'Timing runs=%d  perf repetitions=%d  kernels/perf-run=%d\n\n' \
    "$RUNS" "$PERF_RUNS" "$KERNEL_ITERS"
printf '%-30s %18s %18s\n' "Metric" "naive" "reorder"
printf '%-30s %18s %18s\n' "------" "-----" "-------"
printf '%-30s %18.3f %18.3f\n' "mean time (ms)" "$naive_mean" "$reorder_mean"
printf '%-30s %18.3f %18.3f\n' "median time (ms)" "$naive_median" "$reorder_median"
printf '%-30s %18.3f %18.3f\n' "timing CV (%)" "$naive_cv" "$reorder_cv"
printf '%-30s %18.3f %18.3f\n' "GFLOP/s" "$naive_gflops" "$reorder_gflops"
printf '%-30s %18s %18.3f\n' "speedup vs naive" "1.000" "$speedup"

print_counter()
{
    local label=$1
    local event=$2
    printf '%-30s %18s %18s\n' "$label" \
        "$(divide "${N[$event]}" "$KERNEL_ITERS" 0)" \
        "$(divide "${R[$event]}" "$KERNEL_ITERS" 0)"
}

print_counter "instructions / convolution" instructions
print_counter "cycles / convolution" cycles
printf '%-30s %18s %18s\n' "IPC" \
    "$(divide "${N[instructions]}" "${N[cycles]}" 3)" \
    "$(divide "${R[instructions]}" "${R[cycles]}" 3)"
print_counter "branches / convolution" branch-instructions
print_counter "branch misses / convolution" branch-misses
printf '%-30s %18s %18s\n' "branch MPKI" \
    "$(multiply_divide "${N[branch-misses]}" 1000 "${N[instructions]}" 3)" \
    "$(multiply_divide "${R[branch-misses]}" 1000 "${R[instructions]}" 3)"
print_counter "L1D loads / convolution" L1-dcache-loads
print_counter "L1D load misses / convolution" L1-dcache-load-misses
printf '%-30s %18s %18s\n' "L1D MPKI" \
    "$(multiply_divide "${N[L1-dcache-load-misses]}" 1000 "${N[instructions]}" 3)" \
    "$(multiply_divide "${R[L1-dcache-load-misses]}" 1000 "${R[instructions]}" 3)"
print_counter "cache references / conv" cache-references
print_counter "cache misses / convolution" cache-misses
printf '%-30s %18s %18s\n' "cache-miss MPKI" \
    "$(multiply_divide "${N[cache-misses]}" 1000 "${N[instructions]}" 3)" \
    "$(multiply_divide "${R[cache-misses]}" 1000 "${R[instructions]}" 3)"
print_counter "frontend-stall cycles / conv" stalled-cycles-frontend
print_counter "backend-stall cycles / conv" stalled-cycles-backend
print_counter "stores dispatched / conv" ls_dispatch.store_dispatch
