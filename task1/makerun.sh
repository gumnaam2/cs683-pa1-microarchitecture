make
CORENO=21
echo "Running on core $CORENO"

#reproducible result everytime - turbo will mess with it depending on what else the system is running
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
echo performance | sudo tee /sys/devices/system/cpu/cpu$CORENO/cpufreq/scaling_governor > /dev/null
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct > /dev/null

taskset -c $CORENO perf stat -e cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions ./bin/conv

#reset settings to max performance for my laptop
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/min_perf_pct > /dev/null 
echo 100 | sudo tee /sys/devices/system/cpu/intel_pstate/max_perf_pct > /dev/null