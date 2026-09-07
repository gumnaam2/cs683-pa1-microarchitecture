#!/usr/bin/env bash
# sweep_metrics.sh -- sweeps matrix size only, crossed with
# HW-prefetch on/off, and writes results to CSV.

set -u

CORENO=0

PERF_EVENTS="LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/"

# ---- sweep matrix sizes only -----------------------------------------------
SIZES=(256 512 1024 2048)

CSV_OUT="sweep_results_B64.csv"

OFF_LOG=$(mktemp)
ON_LOG=$(mktemp)
OFF_STDOUT_LOG=$(mktemp)
ON_STDOUT_LOG=$(mktemp)

make

echo "================== Running on core $CORENO =================="
sudo sysctl kernel.perf_event_paranoid=-1

# reproducible result everytime - turbo will mess with it depending
# on what else the system is running
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null

sudo cpupower -c 0 frequency-set -d 4.2GHz -u 4.2GHz

sleep 5

echo "CPU cur freq:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cpupower -c 0 frequency-info


# ---- extraction helpers -----------------------------------------------------

extract() {
    local log="$1"
    local event="$2"

    grep -E "cpu_core/${event}/" "$log" |
        head -n1 |
        sed -E 's/^\s*([0-9,]+)\s+.*/\1/'
}


l2_rate() {
    local miss="$1"
    local ref="$2"
    local miss_n ref_n

    miss_n=$(echo "$miss" | tr -d ',')
    ref_n=$(echo "$ref" | tr -d ',')

    if [ -n "$miss_n" ] &&
       [ -n "$ref_n" ] &&
       [ "$ref_n" -gt 0 ] 2>/dev/null; then

        awk -v m="$miss_n" -v r="$ref_n" \
            'BEGIN { printf "%.2f", (m/r)*100 }'
    else
        echo "n/a"
    fi
}


# Average harness time across perf -r repeats.
# Time is always third-from-last:
#
# stage    correct    time(ms)    GFLOP/s    speedup
#
avg_time_ms() {
    local log="$1"
    local stage="$2"

    grep -E "^${stage}[[:space:]\(]" "$log" |
        awk '{
            sum += $(NF-2)
            n++
        }
        END {
            if (n > 0)
                printf "%.4f", sum/n
            else
                print "n/a"
        }'
}


# ---- run one experiment -----------------------------------------------------

run_one() {
    local msr_val="$1"
    local stdout_log="$2"
    local stderr_log="$3"
    local size="$4"

    sudo wrmsr -p $CORENO 0x1A4 "$msr_val"

    taskset -c $CORENO \
        perf stat \
        -r 2 \
        -e "$PERF_EVENTS" \
        ./bin/matmul prefetch "$size" "$size" "$size" \
        > >(tee "$stdout_log") \
        2> >(tee "$stderr_log" >&2)
}


# ---- CSV header -------------------------------------------------------------

echo "size,hwp,naive_ms,prefetch_ms,speedup,l1d_misses,l2_misses,l2_refs,l2_miss_rate_pct,llc_misses,instructions" \
    > "$CSV_OUT"


# =============================================================================
# SWEEP MATRIX SIZE ONLY
# =============================================================================

for SIZE in "${SIZES[@]}"; do

    echo ""
    echo "============================================================"
    echo "Matrix size = $SIZE"
    echo "============================================================"


    # ---- HW prefetch OFF ----------------------------------------------------

    echo "========== size=$SIZE : HW prefetching disabled =========="

    run_one \
        0xF \
        "$OFF_STDOUT_LOG" \
        "$OFF_LOG" \
        "$SIZE"


    # ---- HW prefetch ON -----------------------------------------------------

    echo "========== size=$SIZE : HW prefetching enabled =========="

    run_one \
        0x0 \
        "$ON_STDOUT_LOG" \
        "$ON_LOG" \
        "$SIZE"


    # ---- Extract both HW states ---------------------------------------------

    for STATE in OFF ON; do

        if [ "$STATE" = "OFF" ]; then

            STDOUT_LOG="$OFF_STDOUT_LOG"
            STDERR_LOG="$OFF_LOG"
            HWP_LABEL="off"

        else

            STDOUT_LOG="$ON_STDOUT_LOG"
            STDERR_LOG="$ON_LOG"
            HWP_LABEL="on"

        fi


        # ---- Execution times ------------------------------------------------

        NAIVE_MS=$(avg_time_ms "$STDOUT_LOG" "naive")

        PF_MS=$(avg_time_ms "$STDOUT_LOG" "prefetch")


        # ---- Speedup --------------------------------------------------------

        SPEEDUP="n/a"

        if [ "$NAIVE_MS" != "n/a" ] &&
           [ "$PF_MS" != "n/a" ]; then

            SPEEDUP=$(awk \
                -v n="$NAIVE_MS" \
                -v p="$PF_MS" \
                'BEGIN {
                    if (p > 0)
                        printf "%.4f", n/p
                    else
                        print "n/a"
                }')
        fi


        # ---- Perf metrics ---------------------------------------------------

        L1D_MISS=$(extract \
            "$STDERR_LOG" \
            'L1-dcache-load-misses' |
            tr -d ',')


        L2_MISS=$(extract \
            "$STDERR_LOG" \
            'l2_rqsts\.miss' |
            tr -d ',')


        L2_REF=$(extract \
            "$STDERR_LOG" \
            'l2_rqsts\.references' |
            tr -d ',')


        L2_RATE=$(l2_rate \
            "$L2_MISS" \
            "$L2_REF")


        LLC_MISS=$(extract \
            "$STDERR_LOG" \
            'LLC-load-misses' |
            tr -d ',')


        INSTR=$(extract \
            "$STDERR_LOG" \
            'instructions' |
            tr -d ',')


        # ---- Write CSV ------------------------------------------------------

        echo "${SIZE},${HWP_LABEL},${NAIVE_MS},${PF_MS},${SPEEDUP},${L1D_MISS},${L2_MISS},${L2_REF},${L2_RATE},${LLC_MISS},${INSTR}" \
            >> "$CSV_OUT"

    done

done


# =============================================================================
# RESTORE CLOCK SETTINGS
#
# PRESERVED FROM YOUR ORIGINAL SCRIPT
# =============================================================================

echo 0 | sudo tee \
    /sys/devices/system/cpu/intel_pstate/no_turbo \
    > /dev/null

sudo cpupower \
    -c 0 \
    frequency-set \
    -g performance

cpupower -c 0 frequency-info


# =============================================================================
# RESULTS
# =============================================================================

echo ""

echo "================== Sweep complete: $CSV_OUT =================="

column -s, -t "$CSV_OUT" | head -n 20

echo "... ($(wc -l < "$CSV_OUT") total rows including header)"


# ---- cleanup ----------------------------------------------------------------

rm -f \
    "$OFF_LOG" \
    "$ON_LOG" \
    "$OFF_STDOUT_LOG" \
    "$ON_STDOUT_LOG"

# #!/usr/bin/env bash
# # sweep_metrics.sh -- sweeps matrix size, prefetch distance, and cache fill
# # level (crossed with HW-prefetch on/off) for Task 2A. Every individual
# # perf stat / wrmsr / cpupower command is UNCHANGED from extract_metrics.sh --
# # this only adds an outer loop over parameter combinations and CSV output.
# # The pre-existing single-run table print is preserved for the default
# # combination (distance=128/32, fill=T1) so old behavior/output isn't lost.

# set -u

# CORENO=0
# PERF_EVENTS="LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/"

# # ---- sweep grid (edit these to change scope) --------------------------------
# SIZES=(1024 2048)
# # L2_DISTANCES=(64 32 16)
# # FILL_LEVELS=(2 3)   # 0=T0 1=T1 2=T2 3=NTA -- matches matmul_prefetch.cpp
# # L1_DISTANCES=(8 16 32)

# CSV_OUT="sweep_results.csv"

# OFF_LOG=$(mktemp)
# ON_LOG=$(mktemp)
# OFF_STDOUT_LOG=$(mktemp)
# ON_STDOUT_LOG=$(mktemp)

# make

# echo "================== Running on core $CORENO =================="
# sudo sysctl kernel.perf_event_paranoid=-1

# # reproducible result everytime - turbo will mess with it depending on what else the system is running
# echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
# sudo cpupower -c 0 frequency-set -d 4.2GHz -u 4.2GHz

# sleep 5

# echo "CPU cur freq:"
# cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
# cpupower -c 0 frequency-info

# # ---- extraction helpers (identical logic to extract_metrics.sh) -------------
# extract() {
#     local log="$1"
#     local event="$2"
#     grep -E "cpu_core/${event}/" "$log" | head -n1 | sed -E 's/^\s*([0-9,]+)\s+.*/\1/'
# }

# l2_rate() {
#     local miss="$1"
#     local ref="$2"
#     local miss_n ref_n
#     miss_n=$(echo "$miss" | tr -d ',')
#     ref_n=$(echo "$ref" | tr -d ',')
#     if [ -n "$miss_n" ] && [ -n "$ref_n" ] && [ "$ref_n" -gt 0 ] 2>/dev/null; then
#         awk -v m="$miss_n" -v r="$ref_n" 'BEGIN { printf "%.2f", (m/r)*100 }'
#     else
#         echo "n/a"
#     fi
# }

# # harness "time(ms)" for a given stage name, averaged across the -r repeats'
# # printed table lines. Indexes from the END of the line (NF-2) rather than
# # a fixed column from the start: "naive (ref)" is two whitespace-separated
# # fields (the stage name contains a space), which shifts every subsequent
# # column right by one compared to "prefetch" -- a fixed $3 reads "yes" (a
# # string) for the naive row and silently sums as 0. speedup and GFLOP/s
# # are always the last two fields regardless of stage-name word count, so
# # time(ms) is reliably the third-from-last field.
# avg_time_ms() {
#     local log="$1"
#     local stage="$2"
#     grep -E "^${stage}[[:space:]\(]" "$log" | awk '{
#         sum += $(NF-2); n++
#     } END {
#         if (n > 0) printf "%.4f", sum/n
#         else print "n/a"
#     }'
# }

# run_one() {
#     local msr_val="$1"
#     local stdout_log="$2"
#     local stderr_log="$3"
#     local size="$4"
#     sudo wrmsr -p $CORENO 0x1A4 "$msr_val"
#     taskset -c $CORENO perf stat -r 2 -e "$PERF_EVENTS" ./bin/matmul prefetch "$size" "$size" "$size" \
#         > >(tee "$stdout_log") 2> >(tee "$stderr_log" >&2)
# }

# echo "size,l2_distance,l1_distance,fill_level,hwp,naive_ms,prefetch_ms,speedup,l1d_misses,l2_misses,l2_refs,l2_miss_rate_pct,llc_misses,instructions" > "$CSV_OUT"

# for SIZE in "${SIZES[@]}"; do
#   for L2D in "${L2_DISTANCES[@]}"; do
#     for L1D in "${L1_DISTANCES[@]}"; do
#     [ "$L1D" -lt 8 ] && L1D=8
#     for FL in "${FILL_LEVELS[@]}"; do
#       export PF_L2_DISTANCE=$L2D PF_L1_DISTANCE=$L1D PF_FILL_LEVEL=$FL

#       echo "========== size=$SIZE l2_dist=$L2D l1_dist=$L1D fill=$FL : HW prefetching disabled =========="
#       run_one 0xF "$OFF_STDOUT_LOG" "$OFF_LOG" "$SIZE"

#       echo "========== size=$SIZE l2_dist=$L2D l1_dist=$L1D fill=$FL : HW prefetching enabled =========="
#       run_one 0x0 "$ON_STDOUT_LOG" "$ON_LOG" "$SIZE"

#       for STATE in OFF ON; do
#         if [ "$STATE" = "OFF" ]; then
#           STDOUT_LOG="$OFF_STDOUT_LOG"; STDERR_LOG="$OFF_LOG"; HWP_LABEL="off"
#         else
#           STDOUT_LOG="$ON_STDOUT_LOG"; STDERR_LOG="$ON_LOG"; HWP_LABEL="on"
#         fi

#         NAIVE_MS=$(avg_time_ms "$STDOUT_LOG" "naive")
#         PF_MS=$(avg_time_ms "$STDOUT_LOG" "prefetch")
#         SPEEDUP="n/a"
#         if [ "$NAIVE_MS" != "n/a" ] && [ "$PF_MS" != "n/a" ]; then
#           SPEEDUP=$(awk -v n="$NAIVE_MS" -v p="$PF_MS" 'BEGIN { if (p>0) printf "%.4f", n/p; else print "n/a" }')
#         fi

#         L1D_MISS=$(extract "$STDERR_LOG" 'L1-dcache-load-misses' | tr -d ',')
#         L2_MISS=$(extract "$STDERR_LOG" 'l2_rqsts\.miss' | tr -d ',')
#         L2_REF=$(extract "$STDERR_LOG" 'l2_rqsts\.references' | tr -d ',')
#         L2_RATE=$(l2_rate "$L2_MISS" "$L2_REF")
#         LLC_MISS=$(extract "$STDERR_LOG" 'LLC-load-misses' | tr -d ',')
#         INSTR=$(extract "$STDERR_LOG" 'instructions' | tr -d ',')

#         echo "${SIZE},${L2D},${L1D},${FL},${HWP_LABEL},${NAIVE_MS},${PF_MS},${SPEEDUP},${L1D_MISS},${L2_MISS},${L2_REF},${L2_RATE},${LLC_MISS},${INSTR}" >> "$CSV_OUT"
#       done
#     done
#   done
# done
# done

# # reset turbo settings for performance for my laptop
# echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
# sudo cpupower -c 0 frequency-set -g performance
# cpupower -c 0 frequency-info

# echo ""
# echo "================== Sweep complete: $CSV_OUT =================="
# column -s, -t "$CSV_OUT" | head -n 20
# echo "... ($(wc -l < "$CSV_OUT") total rows including header)"

# rm -f "$OFF_LOG" "$ON_LOG" "$OFF_STDOUT_LOG" "$ON_STDOUT_LOG"