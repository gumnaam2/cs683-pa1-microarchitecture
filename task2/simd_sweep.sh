#!/usr/bin/env bash
# sweep_simd.sh -- sweeps matrix size x SIMD width for Task 2B (Table 2.1).
# For each (size, width) combination, records naive and simd instruction
# counts, execution times, and speedup normalized to no-SIMD (naive).
#
# SIMD width is selected via the SIMD_WIDTH env var read by matmul_simd.cpp
# at static-init time -- no harness/main.cpp changes needed. Call format
# matches the harness exactly: ./bin/matmul simd M N K

set -u

CORENO=0
SIZES=(4096)
WIDTHS=(128 256)

CSV_OUT="simd_sweep_results.csv"

make

echo "================== Running on core $CORENO =================="
sudo sysctl kernel.perf_event_paranoid=-1
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
sudo cpupower -c 0 frequency-set -d 4.2GHz -u 4.2GHz
sleep 5
echo "CPU cur freq:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cpupower -c 0 frequency-info

# harness "time(ms)" for a given stage, from a single run's stdout table.
# Indexes from the END of the line (NF-2) since "naive (ref)" is two
# whitespace-separated fields -- a fixed column position would misread it.
extract_time_ms() {
    local log="$1"
    local stage="$2"
    grep -E "^${stage}[[:space:]\(]" "$log" | head -n1 | awk '{ print $(NF-2) }'
}

# raw instruction count from perf stat's cpu_core-scoped line -- matches
# only cpu_core/instructions/ specifically, since the <not counted>
# cpu_atom/instructions/ line appears first on this hybrid CPU and would
# otherwise be picked up instead by a looser match.
extract_instructions() {
    local log="$1"
    grep -E "cpu_core/instructions/" "$log" | head -n1 | sed -E 's/^\s*([0-9,]+)\s+.*/\1/' | tr -d ','
}

echo "size,simd_width,naive_time_ms,naive_instructions,simd_time_ms,simd_instructions,speedup" > "$CSV_OUT"

for SIZE in "${SIZES[@]}"; do
  for WIDTH in "${WIDTHS[@]}"; do
    export SIMD_WIDTH=$WIDTH

    STDOUT_LOG=$(mktemp)
    STDERR_LOG=$(mktemp)

    echo "========== size=$SIZE width=$WIDTH =========="
    taskset -c $CORENO perf stat -e instructions ./bin/matmul simd "$SIZE" "$SIZE" "$SIZE" \
        > >(tee "$STDOUT_LOG") 2> >(tee "$STDERR_LOG" >&2)

    NAIVE_MS=$(extract_time_ms "$STDOUT_LOG" "naive")
    SIMD_MS=$(extract_time_ms "$STDOUT_LOG" "simd")
    INSTR_TOTAL=$(extract_instructions "$STDERR_LOG")

    # perf stat here reports ONE instruction count for the whole process
    # (both naive and simd stages run in the same invocation) -- there is
    # no way to separate per-stage instruction counts from a single
    # combined perf run. Recorded under both columns with a note; see
    # comment below the loop for how to get true per-stage counts if
    # that granularity is required for the table.
    NAIVE_INSTR="$INSTR_TOTAL"
    SIMD_INSTR="$INSTR_TOTAL"

    SPEEDUP="n/a"
    if [ -n "$NAIVE_MS" ] && [ -n "$SIMD_MS" ]; then
        SPEEDUP=$(awk -v n="$NAIVE_MS" -v s="$SIMD_MS" 'BEGIN { if (s>0) printf "%.4f", n/s; else print "n/a" }')
    fi

    echo "${SIZE},${WIDTH},${NAIVE_MS},${NAIVE_INSTR},${SIMD_MS},${SIMD_INSTR},${SPEEDUP}" >> "$CSV_OUT"

    rm -f "$STDOUT_LOG" "$STDERR_LOG"
  done
done

echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
sudo cpupower -c 0 frequency-set -g performance
cpupower -c 0 frequency-info

echo ""
echo "================== Sweep complete: $CSV_OUT =================="
column -s, -t "$CSV_OUT"

# NOTE on per-stage instruction counts:
# perf stat measures the whole ./bin/matmul process, which runs BOTH the
# naive and simd stages in one invocation -- so "instructions" above is
# their combined total, not two separate numbers. If the table strictly
# needs naive-only vs simd-only instruction counts, the cleanest fix is
# a harness flag to run a single named stage in isolation (e.g.
# `./bin/matmul simd-only M N K`), then run perf stat twice per
# (size, width) -- once with that flag targeting naive, once targeting
# simd -- and fill NAIVE_INSTR / SIMD_INSTR from those two separate runs
# instead of the one combined figure used here.