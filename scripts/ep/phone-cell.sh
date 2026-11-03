#!/system/bin/sh
# Phone side of one expert-parallel run (Pixel 7 / 7 Pro, Tensor G2), as root.
# Pins clocks (optional), samples clocks and temperatures, snapshots ODPM rail energy, runs CMD, restores clocks.
#
#   OUT=dir CMD="..." [CPUS=f0] [PIN=p0/p4/p6/gpu/mif kHz, e.g. 1328000/1491000/1426000/701000/1539000]
#   [SAMPLE_MS=100] sh phone-cell.sh
# Writes OUT/{cmd.txt,stdout.log,stderr.log,exit_code,freq.csv,odpm-before.txt,odpm-after.txt,settings-*.txt}
set -u
: "${OUT:?}" "${CMD:?}"
CPUS=${CPUS:-f0}
SAMPLE_MS=${SAMPLE_MS:-100}
PIN=${PIN:-}
MALI=/sys/devices/platform/28000000.mali
MIF=/sys/class/devfreq/17000010.devfreq_mif
CF=/sys/devices/system/cpu/cpufreq
mkdir -p "$OUT"
echo "$CMD" > "$OUT/cmd.txt"

settings() {
    for p in 0 4 6; do echo "policy$p gov=$(cat $CF/policy$p/scaling_governor) min=$(cat $CF/policy$p/scaling_min_freq) max=$(cat $CF/policy$p/scaling_max_freq)"; done
    echo "gpu min=$(cat $MALI/scaling_min_freq) max=$(cat $MALI/scaling_max_freq) policy=$(cat $MALI/power_policy)"
    echo "mif min=$(cat $MIF/min_freq) max=$(cat $MIF/max_freq)"
    for z in /sys/class/thermal/thermal_zone*; do echo "thermal $(cat $z/type)=$(cat $z/temp)"; done
}
odpm() {
    for d in 0 1; do cat /sys/bus/iio/devices/iio:device$d/energy_value; done
}
settings > "$OUT/settings-before.txt"

SAVE=$OUT/.restore
: > "$SAVE"
for p in 0 4 6; do
    echo "echo $(cat $CF/policy$p/scaling_governor) > $CF/policy$p/scaling_governor" >> "$SAVE"
    echo "echo $(cat $CF/policy$p/cpuinfo_min_freq) > $CF/policy$p/scaling_min_freq" >> "$SAVE"
    echo "echo $(cat $CF/policy$p/scaling_max_freq) > $CF/policy$p/scaling_max_freq" >> "$SAVE"
    echo "echo $(cat $CF/policy$p/scaling_min_freq) > $CF/policy$p/scaling_min_freq" >> "$SAVE"
done
echo "echo $(cat $MALI/scaling_max_freq) > $MALI/scaling_max_freq; echo $(cat $MALI/scaling_min_freq) > $MALI/scaling_min_freq" >> "$SAVE"
echo "echo $(cat $MIF/min_freq) > $MIF/min_freq; echo $(cat $MIF/max_freq) > $MIF/max_freq; echo $(cat $MIF/min_freq) > $MIF/min_freq" >> "$SAVE"
restore() { sh "$SAVE"; settings > "$OUT/settings-after.txt"; }
trap restore EXIT

pin_to() { # file_min file_max kHz
    echo "$3" > "$2"; echo "$3" > "$1"; echo "$3" > "$2"
}
if [ -n "$PIN" ]; then
    IFS=/ read f0 f4 f6 fg fm <<EOF
$PIN
EOF
    for pf in "0 $f0" "4 $f4" "6 $f6"; do
        set -- $pf
        echo performance > $CF/policy$1/scaling_governor
        pin_to $CF/policy$1/scaling_min_freq $CF/policy$1/scaling_max_freq "$2"
    done
    pin_to $MALI/scaling_min_freq $MALI/scaling_max_freq "$fg"
    pin_to $MIF/min_freq $MIF/max_freq "$fm"
    settings > "$OUT/settings-pinned.txt"
fi

# sampler on the little cores; clock: CLOCK_MONOTONIC via /proc/uptime is too coarse, use date (realtime ns)
(
    echo "t_real_ns,cpu0,cpu4,cpu6,gpu,gpu_util,mif,big_mC"
    while :; do
        big=0
        for z in /sys/class/thermal/thermal_zone*; do [ "$(cat $z/type)" = BIG ] && big=$(cat $z/temp); done
        echo "$(date +%s%N),$(cat $CF/policy0/scaling_cur_freq),$(cat $CF/policy4/scaling_cur_freq),$(cat $CF/policy6/scaling_cur_freq),$(cat $MALI/cur_freq),$(cat $MALI/utilization),$(cat $MIF/cur_freq),$big"
        usleep $((SAMPLE_MS * 1000))
    done
) > "$OUT/freq.csv" 2>/dev/null &
SAMPLER=$!
taskset -p 0f $SAMPLER > /dev/null 2>&1

odpm > "$OUT/odpm-before.txt"
echo "$(date +%s%N)" > "$OUT/t_start_real_ns"
taskset $CPUS sh -c "$CMD" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
echo $? > "$OUT/exit_code"
echo "$(date +%s%N)" > "$OUT/t_end_real_ns"
odpm > "$OUT/odpm-after.txt"
kill $SAMPLER 2> /dev/null
wait $SAMPLER 2> /dev/null
exit 0
