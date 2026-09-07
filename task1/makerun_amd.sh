#!/usr/bin/env bash
set -euo pipefail

CORENO=${CORENO:-2}
BENCH=(./bin/conv)
RUNS=${4:-${RUNS:-5}}
SEED=${SEED:-1234}
MODE=${MODE:-fixed}
FIXED_FREQ_KHZ=${FIXED_FREQ_KHZ:-2000000}

usage()
{
    cat <<EOF
Usage: $0 [H W K [runs]]

Runs naive, reorder, unroll, tile, simd, and optimized using ./bin/conv,
then reports the average speedup over multiple independent harness runs.

Defaults: H=2048, W=2048, K=3, runs=5
Environment overrides: CORENO, RUNS, SEED, MODE, FIXED_FREQ_KHZ
EOF
}

if (( $# == 1 )) && [[ $1 == "-h" || $1 == "--help" ]]; then
    usage
    exit 0
fi

if (( $# == 0 )); then
    H=2048
    W=2048
    K=3
elif (( $# == 3 || $# == 4 )); then
    H=$1
    W=$2
    K=$3
else
    usage >&2
    exit 2
fi

for value in "$H" "$W" "$K" "$RUNS" "$SEED" "$CORENO"; do
    if [[ ! $value =~ ^[0-9]+$ ]]; then
        echo "ERROR: H, W, K, runs, SEED, and CORENO must be integers." >&2
        exit 2
    fi
done

if (( H <= 0 || W <= 0 || K <= 0 || RUNS <= 0 )); then
    echo "ERROR: H, W, K, and runs must be positive." >&2
    exit 2
fi
if (( W % 8 != 0 )); then
    echo "ERROR: W must be a multiple of 8, as required by ./bin/conv." >&2
    exit 2
fi
if (( K % 2 == 0 )); then
    echo "ERROR: K must be odd, as required by ./bin/conv." >&2
    exit 2
fi

make

echo "Running on logical CPU $CORENO"
echo "Workload: H=$H W=$W K=$K seed=$SEED"
echo "Independent harness runs: $RUNS"

# ------------------------------------------------------------
# Locate this CPU's amd-pstate policy
# ------------------------------------------------------------

POLICY=$(readlink -f "/sys/devices/system/cpu/cpu${CORENO}/cpufreq")
SIBLING_LIST=$(cat "/sys/devices/system/cpu/cpu${CORENO}/topology/thread_siblings_list")
SIBLING=$(printf '%s\n' "$SIBLING_LIST" | tr ',' '\n' | awk -v cpu="$CORENO" '$1 != cpu { print; exit }')

echo "CPUFreq policy: $POLICY"
echo "Driver: $(cat "$POLICY/scaling_driver")"
echo "SMT siblings: $SIBLING_LIST"
echo "Frequency mode: $MODE"

if [[ "$MODE" != fixed && "$MODE" != performance ]]; then
    echo "ERROR: MODE must be either 'fixed' or 'performance'."
    exit 1
fi

if [[ -z "$SIBLING" ]]; then
    echo "ERROR: Cannot determine the SMT sibling of CPU $CORENO."
    exit 1
fi

SIBLING_ONLINE_FILE="/sys/devices/system/cpu/cpu${SIBLING}/online"
if [[ ! -e "$SIBLING_ONLINE_FILE" ]]; then
    echo "ERROR: Cannot control SMT sibling CPU $SIBLING."
    exit 1
fi

# ------------------------------------------------------------
# Save original settings
# ------------------------------------------------------------

OLD_GOVERNOR=$(cat "$POLICY/scaling_governor")
OLD_MIN_FREQ=$(cat "$POLICY/scaling_min_freq")
OLD_MAX_FREQ=$(cat "$POLICY/scaling_max_freq")
CPUINFO_MIN_FREQ=$(cat "$POLICY/cpuinfo_min_freq")
CPUINFO_MAX_FREQ=$(cat "$POLICY/cpuinfo_max_freq")
OLD_SIBLING_ONLINE=$(cat "$SIBLING_ONLINE_FILE")

# AMD boost can appear in either of these locations depending on kernel.
if [[ -e /sys/devices/system/cpu/cpufreq/boost ]]; then
    BOOST_FILE=/sys/devices/system/cpu/cpufreq/boost
elif [[ -e "$POLICY/boost" ]]; then
    BOOST_FILE="$POLICY/boost"
else
    echo "ERROR: Cannot find AMD boost control."
    exit 1
fi

OLD_BOOST=$(cat "$BOOST_FILE")

# Save EPP if amd-pstate-epp exposes it.
if [[ -e "$POLICY/energy_performance_preference" ]]; then
    OLD_EPP=$(cat "$POLICY/energy_performance_preference")
else
    OLD_EPP=""
fi


# ------------------------------------------------------------
# Restore everything automatically, even on Ctrl+C
# ------------------------------------------------------------

cleanup()
{
    status=$?
    trap - EXIT INT TERM
    set +e

    echo
    echo "Restoring CPU settings..."

    echo "$OLD_BOOST" |
        sudo tee "$BOOST_FILE" >/dev/null ||
        echo "WARNING: Could not restore boost." >&2

    echo "$OLD_MAX_FREQ" |
        sudo tee "$POLICY/scaling_max_freq" >/dev/null ||
        echo "WARNING: Could not restore scaling_max_freq." >&2

    echo "$OLD_MIN_FREQ" |
        sudo tee "$POLICY/scaling_min_freq" >/dev/null ||
        echo "WARNING: Could not restore scaling_min_freq." >&2

    echo "$OLD_GOVERNOR" |
        sudo tee "$POLICY/scaling_governor" >/dev/null ||
        echo "WARNING: Could not restore scaling governor." >&2

    if [[ -n "$OLD_EPP" ]]; then
        echo "$OLD_EPP" |
            sudo tee "$POLICY/energy_performance_preference" >/dev/null ||
            echo "WARNING: Could not restore EPP." >&2
    fi

    if [[ "$OLD_SIBLING_ONLINE" == 1 ]]; then
        echo 1 |
            sudo tee "$SIBLING_ONLINE_FILE" >/dev/null ||
            echo "WARNING: Could not bring CPU $SIBLING back online." >&2
    fi

    if [[ -n ${RESULTS_FILE:-} ]]; then
        rm -f -- "$RESULTS_FILE"
    fi

    echo "Settings restored."

    exit "$status"
}

sudo -v

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Suspending here would leave boost, CPU online state, frequency limits, and
# perf permissions modified because EXIT cleanup does not run on Ctrl+Z.
trap '' TSTP


# ------------------------------------------------------------
# Keep the benchmark's physical core free of SMT contention
# ------------------------------------------------------------

if [[ "$OLD_SIBLING_ONLINE" == 1 ]]; then
    echo 0 |
        sudo tee "$SIBLING_ONLINE_FILE" >/dev/null
    echo "SMT sibling CPU $SIBLING taken offline."
else
    echo "SMT sibling CPU $SIBLING is already offline."
fi


# ------------------------------------------------------------
# Disable AMD Core Performance Boost in fixed-frequency mode
# ------------------------------------------------------------

if [[ "$MODE" == fixed ]]; then
    echo 0 |
        sudo tee "$BOOST_FILE" >/dev/null
    echo "AMD boost disabled."
else
    echo "AMD boost left at its original setting ($OLD_BOOST)."
fi


# ------------------------------------------------------------
# Use performance governor
# ------------------------------------------------------------

if grep -qw performance "$POLICY/scaling_available_governors"; then
    echo performance |
        sudo tee "$POLICY/scaling_governor" >/dev/null
fi


# amd-pstate EPP: bias selected CPU toward deterministic performance
if [[ -e "$POLICY/energy_performance_preference" ]] &&
   grep -qw performance "$POLICY/energy_performance_available_preferences"; then

    echo performance |
        sudo tee "$POLICY/energy_performance_preference" >/dev/null
fi


# ------------------------------------------------------------
# Optionally request one fixed, non-boost frequency
# ------------------------------------------------------------

if [[ "$MODE" == fixed ]]; then
    if (( FIXED_FREQ_KHZ < CPUINFO_MIN_FREQ || FIXED_FREQ_KHZ > CPUINFO_MAX_FREQ )); then
        echo "ERROR: Fixed frequency ${FIXED_FREQ_KHZ} kHz is outside the supported range."
        exit 1
    fi

    echo "$FIXED_FREQ_KHZ" |
        sudo tee "$POLICY/scaling_max_freq" >/dev/null
    echo "$FIXED_FREQ_KHZ" |
        sudo tee "$POLICY/scaling_min_freq" >/dev/null

    APPLIED_MIN=$(cat "$POLICY/scaling_min_freq")
    APPLIED_MAX=$(cat "$POLICY/scaling_max_freq")
    if [[ "$APPLIED_MIN" != "$FIXED_FREQ_KHZ" || "$APPLIED_MAX" != "$FIXED_FREQ_KHZ" ]]; then
        echo "ERROR: Failed to apply the requested fixed frequency."
        exit 1
    fi

    echo "Frequency requested at ${FIXED_FREQ_KHZ} kHz."
fi


# ------------------------------------------------------------
# Repeated all-stage wall-clock benchmark
# ------------------------------------------------------------

echo
echo "Wall-clock benchmark"
RESULTS_FILE=$(mktemp)

for ((run = 1; run <= RUNS; ++run)); do
    echo
    echo "Run $run of $RUNS"
    output=$(
        taskset -c "$CORENO" \
            setarch "$(uname -m)" -R \
            "${BENCH[@]}" all "$H" "$W" "$K" "$SEED"
    )
    printf '%s\n' "$output"

    if printf '%s\n' "$output" |
       grep -Eq '^(naive \(ref\)|reorder|unroll|tile|simd|optimized)[[:space:]]+NO'; then
        echo "ERROR: At least one implementation failed correctness in run $run." >&2
        exit 1
    fi

    printf '%s\n' "$output" |
        awk -v run="$run" '
            $1 == "naive" || $1 == "reorder" || $1 == "unroll" ||
            $1 == "tile" || $1 == "simd" || $1 == "optimized" {
                speedup = $NF
                sub(/x$/, "", speedup)
                print run, $1, speedup
            }
        ' >> "$RESULTS_FILE"
done

echo
echo "Average speedup for H=$H W=$W K=$K ($RUNS runs)"
awk '
    BEGIN {
        split("naive reorder unroll tile simd optimized", order)
        printf "%-14s %7s %12s %10s %10s %10s\n", \
               "stage", "runs", "avg speedup", "stddev", "min", "max"
        printf "--------------------------------------------------------------------\n"
    }
    {
        stage = $2
        value = $3 + 0
        sum[stage] += value
        sumsq[stage] += value * value
        count[stage]++
        if (!(stage in minimum) || value < minimum[stage]) minimum[stage] = value
        if (!(stage in maximum) || value > maximum[stage]) maximum[stage] = value
    }
    END {
        for (i = 1; i <= 6; ++i) {
            stage = order[i]
            if (count[stage] == 0) {
                printf "ERROR: no result parsed for %s\n", stage > "/dev/stderr"
                exit 1
            }
            average = sum[stage] / count[stage]
            variance = sumsq[stage] / count[stage] - average * average
            if (variance < 0) variance = 0
            printf "%-14s %7d %11.3fx %9.3f %9.2fx %9.2fx\n", \
                   stage, count[stage], average, sqrt(variance), \
                   minimum[stage], maximum[stage]
        }
    }
' "$RESULTS_FILE"
