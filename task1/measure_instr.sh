#!/usr/bin/env bash

set -u

REPS=100
BINARY="./bin/num_instr"
IMPLEMENTATIONS=("naive" "128" "256" "512")

# Store results
declare -A TOTAL_INSTRUCTIONS
declare -A AVG_INSTRUCTIONS

# Make perf output predictable for parsing
export LC_ALL=C

# Check that benchmark exists
if [[ ! -x "$BINARY" ]]; then
    echo "Error: $BINARY does not exist or is not executable."
    echo "Run 'make' first."
    exit 1
fi

# Ask for sudo password once at the beginning
sudo -v || exit 1

echo "Measuring retired instructions..."
echo

for impl in "${IMPLEMENTATIONS[@]}"; do
    echo "Running implementation: $impl"

    tmpfile=$(mktemp)

    # perf writes its statistics to stderr.
    # -x ';' makes the output easy to parse.
    if ! sudo perf stat \
        -x ';' \
        -e instructions:u,cycles:u \
        -- "$BINARY" "$impl" \
        > /dev/null 2> "$tmpfile"
    then
        echo "  ERROR: perf failed for implementation '$impl'."
        cat "$tmpfile"
        rm -f "$tmpfile"
        echo
        continue
    fi

    # perf CSV-style format:
    # value ; unit ; event ; ...
    instructions=$(awk -F';' '
        $3 ~ /^instructions/ {
            gsub(/[[:space:]]/, "", $1)
            print $1
            exit
        }
    ' "$tmpfile")

    rm -f "$tmpfile"

    if [[ -z "$instructions" || "$instructions" == "<notcounted>" || "$instructions" == "<notsupported>" ]]; then
        echo "  ERROR: Could not obtain instruction count."
        echo
        continue
    fi

    TOTAL_INSTRUCTIONS["$impl"]="$instructions"

    # Divide by the number of convolution repetitions.
    average=$(awk -v total="$instructions" -v reps="$REPS" \
        'BEGIN { printf "%.2f", total / reps }')

    AVG_INSTRUCTIONS["$impl"]="$average"

    printf "  Total instructions:       %'d\n" "$instructions"
    printf "  Instructions / conv:      %'.2f\n" "$average"
    echo
done

echo "=============================================="
echo "Summary"
echo "=============================================="
printf "%-12s %20s %22s\n" \
    "Version" "Total Instructions" "Instructions / Conv"
printf "%-12s %20s %22s\n" \
    "-------" "------------------" "-------------------"

for impl in "${IMPLEMENTATIONS[@]}"; do
    if [[ -n "${TOTAL_INSTRUCTIONS[$impl]:-}" ]]; then
        printf "%-12s %20d %22.2f\n" \
            "$impl" \
            "${TOTAL_INSTRUCTIONS[$impl]}" \
            "${AVG_INSTRUCTIONS[$impl]}"
    else
        printf "%-12s %20s %22s\n" \
            "$impl" "FAILED" "FAILED"
    fi
done