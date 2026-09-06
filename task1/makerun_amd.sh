#!/usr/bin/env bash
set -euo pipefail

CORENO=4

BENCH=(./bin/conv)
RUNS=5

echo "Running on logical CPU $CORENO"

# ------------------------------------------------------------
# Locate this CPU's amd-pstate policy
# ------------------------------------------------------------

POLICY=$(readlink -f "/sys/devices/system/cpu/cpu${CORENO}/cpufreq")

echo "CPUFreq policy: $POLICY"
echo "Driver: $(cat "$POLICY/scaling_driver")"
echo "SMT siblings: $(cat "/sys/devices/system/cpu/cpu${CORENO}/topology/thread_siblings_list")"

# ------------------------------------------------------------
# Save original settings
# ------------------------------------------------------------

OLD_PARANOID=$(sysctl -n kernel.perf_event_paranoid)
OLD_GOVERNOR=$(cat "$POLICY/scaling_governor")

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
    echo
    echo "Restoring CPU settings..."

    echo "$OLD_BOOST" |
        sudo tee "$BOOST_FILE" >/dev/null

    echo "$OLD_GOVERNOR" |
        sudo tee "$POLICY/scaling_governor" >/dev/null

    if [[ -n "$OLD_EPP" ]]; then
        echo "$OLD_EPP" |
            sudo tee "$POLICY/energy_performance_preference" >/dev/null || true
    fi

    sudo sysctl -q -w kernel.perf_event_paranoid="$OLD_PARANOID"

    echo "Settings restored."
}

trap cleanup EXIT INT TERM


# ------------------------------------------------------------
# Allow perf
# ------------------------------------------------------------

sudo sysctl -q -w kernel.perf_event_paranoid=-1


# ------------------------------------------------------------
# Disable AMD Core Performance Boost
# ------------------------------------------------------------

echo 0 |
    sudo tee "$BOOST_FILE" >/dev/null

echo "AMD boost disabled."


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
# Benchmark
# ------------------------------------------------------------

perf stat -r "$RUNS" \
    -e cache-misses \
    -e L1-dcache-loads \
    -e L1-dcache-load-misses \
    -e L1-dcache-stores \
    -e L1-dcache-store-misses \
    -e L1-icache-loads \
    -e L1-icache-load-misses \
    -e instructions \
    -- \
    setarch "$(uname -m)" -R \
    taskset -c "$CORENO" "${BENCH[@]}"