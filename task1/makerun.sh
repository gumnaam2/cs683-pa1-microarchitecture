make
CORENO=0
echo "Running on core $CORENO"
sudo sysctl kernel.perf_event_paranoid=-1

#reproducible result everytime - turbo will mess with it depending on what else the system is running
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null

taskset -c $CORENO perf stat -e cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions ./bin/conv

#reset turbo settings for performance for my laptop
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null