make
CORENO=0
echo "================== Running on core $CORENO =================="
sudo sysctl kernel.perf_event_paranoid=-1

#reproducible result everytime - turbo will mess with it depending on what else the system is running
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
sudo cpupower -c 0 frequency-set -d 4.2GHz-u 4.2GHz

sleep 5

echo "CPU cur freq:" 
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cpupower -c 0 frequency-info

# echo 0 | sudo tee /sys/devices/system/cpu/cpu5/online #offline logical core that shares physical core
#doesn't work
# sudo cpupower frequency-set -c $CORENO -f 4500MHz

#disable HW prefetcher
echo "========== HW prefetching disabled =========="
sudo wrmsr -p $CORENO 0x1A4 0xF
echo "Read value of MSR 0x1A4:"
sudo rdmsr -p $CORENO 0x1A4
taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss ./bin/matmul
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss ./bin/matmul optimized
# taskset -c $CORENO perf stat -r 2 -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 512 512 512
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul naive 1024 1024 1024
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 256 256 256
# taskset -c $CORENO perf stat -r 2 -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 2048 2048 2048
# taskset -c $CORENO perf stat -r 2 -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 4096 4096 4096
# taskset -c $CORENO perf stat -e cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions ./bin/matmul prefetch 4096 4096 4096

echo "========== HW prefetching enabled =========="
sudo wrmsr -p $CORENO 0x1A4 0x0
echo "Read value of MSR 0x1A4:"
sudo rdmsr -p $CORENO 0x1A4
taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul
# taskset -c $CORENO perf stat -r 2 -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 512 512 512
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul naive 1024 1024 1024
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 256 256 256
# taskset -c $CORENO perf stat -r 2 -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 2048 2048 2048
# taskset -c $CORENO perf stat -r 2 -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss,cpu_core/l2_rqsts.miss/,cpu_core/l2_rqsts.references/ ./bin/matmul prefetch 4096 4096 4096
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss ./bin/matmul optimized
# taskset -c $CORENO perf stat -e LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions,cpu_core/sw_prefetch_access.t0/,l1d_pend_miss.fb_full,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,cycle_activity.stalls_l1d_miss ./bin/matmul
# taskset -c $CORENO perf stat -e cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions ./bin/matmul prefetch 4096 4096 4096

# CORENO=21
# echo "================== Running on core $CORENO =================="

# #disable HW prefetcher
# echo "========== HW prefetching disabled =========="
# sudo wrmsr -a 0x1A4 0xf
# taskset -c $CORENO perf stat -e cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions ./bin/matmul prefetch

# echo "========== HW prefetching enabled =========="
# sudo wrmsr -a 0x1A4 0
# taskset -c $CORENO perf stat -e cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,instructions ./bin/matmul prefetch

#reset turbo settings for performance for my laptop
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null
sudo cpupower -c 0 frequency-set -g performance
cpupower -c 0 frequency-info
# echo 1 | sudo tee /sys/devices/system/cpu/cpu5/online #offline logical core that shares physical core

#doesn't work
# sudo cpupower frequency-set -c $CORENO -g schedutil