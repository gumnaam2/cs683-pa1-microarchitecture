#!/usr/bin/env bash
# extract_metrics.sh -- runs the SAME commands as makerun.sh (unchanged),
# but tees perf's output to files so LLC-load-misses, L1-dcache-load-misses,
# L2 misses, and instructions can be pulled out and shown as a HW-prefetch
# on/off comparison table at the end. Does not alter any existing behavior --
# only adds output capture around the existing perf stat invocations.

set -u

CORENO=0
PERF_EVENTS="LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/"

OFF_LOG=$(mktemp)
ON_LOG=$(mktemp)
OFF_STDOUT_LOG=$(mktemp)
ON_STDOUT_LOG=$(mktemp)

make

echo "================== Running on core $CORENO =================="
sudo sysctl kernel.perf_event_paranoid=-1

# reproducible result everytime - turbo will mess with it depending on what else the system is running
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
sudo cpupower -c 0 frequency-set -d 4.2GHz -u 4.2GHz


sleep 5

echo "CPU cur freq:" 
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cpupower -c 0 frequency-info


# ---- disable HW prefetcher -------------------------------------------------
echo "========== HW prefetching disabled =========="
sudo wrmsr -p $CORENO 0x1A4 0xF
echo "Read value of MSR 0x1A4:"
sudo rdmsr -p $CORENO 0x1A4
taskset -c $CORENO perf stat -r 5 -e "$PERF_EVENTS" ./bin/matmul optimized \
    > >(tee "$OFF_STDOUT_LOG") 2> >(tee "$OFF_LOG" >&2)

# ---- enable HW prefetcher ---------------------------------------------------
echo "========== HW prefetching enabled =========="
sudo wrmsr -p $CORENO 0x1A4 0x0
echo "Read value of MSR 0x1A4:"
sudo rdmsr -p $CORENO 0x1A4
taskset -c $CORENO perf stat -r 5 -e "$PERF_EVENTS" ./bin/matmul optimized \
    > >(tee "$ON_STDOUT_LOG") 2> >(tee "$ON_LOG" >&2)

# reset turbo settings for performance for my laptop
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
sudo cpupower -c 0 frequency-set -g performance
cpupower -c 0 frequency-info

# ---- extract the metrics from each captured log ------------------------------
# Matches only the cpu_core/<event>/ line specifically -- the raw event name
# alone also matches the "<not counted> cpu_atom/<event>/" line above it,
# which appears first and would otherwise be picked up instead.
extract() {
    local log="$1"
    local event="$2"
    grep -E "cpu_core/${event}/" "$log" | head -n1 | sed -E 's/^\s*([0-9,]+)\s+.*/\1/'
}

OFF_LLC=$(extract "$OFF_LOG" 'LLC-load-misses')
OFF_L1D=$(extract "$OFF_LOG" 'L1-dcache-load-misses')
OFF_L2M=$(extract "$OFF_LOG" 'l2_rqsts\.miss')
OFF_L2R=$(extract "$OFF_LOG" 'l2_rqsts\.references')
OFF_INSTR=$(extract "$OFF_LOG" 'instructions')

ON_LLC=$(extract "$ON_LOG" 'LLC-load-misses')
ON_L1D=$(extract "$ON_LOG" 'L1-dcache-load-misses')
ON_L2M=$(extract "$ON_LOG" 'l2_rqsts\.miss')
ON_L2R=$(extract "$ON_LOG" 'l2_rqsts\.references')
ON_INSTR=$(extract "$ON_LOG" 'instructions')

# L2 miss rate (%) computed from raw miss/reference counts, since perf's
# own inline "# X% of ..." annotation isn't guaranteed to appear for every
# derived event pairing the way it does for L1 automatically.
l2_rate() {
    local miss="$1"
    local ref="$2"
    local miss_n ref_n
    miss_n=$(echo "$miss" | tr -d ',')
    ref_n=$(echo "$ref" | tr -d ',')
    if [ -n "$miss_n" ] && [ -n "$ref_n" ] && [ "$ref_n" -gt 0 ] 2>/dev/null; then
        awk -v m="$miss_n" -v r="$ref_n" 'BEGIN { printf "%.2f%%", (m/r)*100 }'
    else
        echo "n/a"
    fi
}

OFF_L2RATE=$(l2_rate "$OFF_L2M" "$OFF_L2R")
ON_L2RATE=$(l2_rate "$ON_L2M" "$ON_L2R")

# ---- average speedup from the harness's own "optimized ... x" lines ---------
# perf stat -r 5 runs the binary 5 times, so its stdout table (and this line)
# appears once per repeat -- average across however many actually printed.
avg_speedup() {
    local log="$1"
    grep -E '^optimized' "$log" | awk '{
        gsub("x","",$NF); sum+=$NF; n++
    } END {
        if (n > 0) printf "%.2fx (avg of %d runs)", sum/n, n
        else print "n/a"
    }'
}

OFF_SPEEDUP=$(avg_speedup "$OFF_STDOUT_LOG")
ON_SPEEDUP=$(avg_speedup "$ON_STDOUT_LOG")

echo ""
echo "================== Metric comparison =================="
printf "%-22s %-20s %-20s\n" "metric" "HW prefetch OFF" "HW prefetch ON"
printf "%-22s %-20s %-20s\n" "L1-dcache-load-misses" "$OFF_L1D"   "$ON_L1D"
printf "%-22s %-20s %-20s\n" "L2 misses"             "$OFF_L2M"   "$ON_L2M"
printf "%-22s %-20s %-20s\n" "L2 references"         "$OFF_L2R"   "$ON_L2R"
printf "%-22s %-20s %-20s\n" "L2 miss rate"          "$OFF_L2RATE" "$ON_L2RATE"
printf "%-22s %-20s %-20s\n" "LLC-load-misses"       "$OFF_LLC"   "$ON_LLC"
printf "%-22s %-20s %-20s\n" "instructions"          "$OFF_INSTR" "$ON_INSTR"
printf "%-22s %-20s %-20s\n" "avg speedup"           "$OFF_SPEEDUP" "$ON_SPEEDUP"

rm -f "$OFF_LOG" "$ON_LOG" "$OFF_STDOUT_LOG" "$ON_STDOUT_LOG"