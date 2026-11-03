#!/usr/bin/env bash
# Host side: push the Android build, scripts and model to the phone, run one cell with phone-cell.sh, pull results.
#
#   NAME=cell-name CMD='$BIN/llama-ep-check -m $MODEL ...' [PIN=p0/p4/p6/gpu/mif] [CPUS=f0] [PUSH=1] bash run-phone.sh
# In CMD, $BIN, $MODEL and $OUT expand on the phone (CMD runs with cwd $OUT). Results land in RESULTS/NAME.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ADB=${ADB:-/mnt/c/Users/Mohamed/Downloads/platform-tools-latest-windows/platform-tools/adb.exe}
SERIAL=${SERIAL:-192.168.0.209:5555}
BUILD=${BUILD:-$HERE/../../build-android-vk}
MODEL_HOST=${MODEL_HOST:-/home/mohamed/mobilemoe/models/MobileMoE-M-QAT-q4_0.gguf}
RESULTS=${RESULTS:-/home/mohamed/mobilemoe/ep-results}
REMOTE=${REMOTE:-/data/local/tmp/ep}
STAGE=/mnt/c/monsoon-build/ep-stage # adb.exe reads Windows paths; files pass through here and are removed
: "${NAME:?}" "${CMD:?}"
adb_() { (cd /mnt/c && "$ADB" -s "$SERIAL" "$@"); }
push_as_root() { # host_file remote_file [mode]
    mkdir -p "$STAGE"
    local tmp="$STAGE/$(basename "$1")"
    cp "$1" "$tmp"
    adb_ push "$(wslpath -w "$tmp")" "$REMOTE/.push" > /dev/null
    rm -f "$tmp"
    adb_ shell "su -c 'cp $REMOTE/.push $2 && chmod ${3:-644} $2 && rm $REMOTE/.push'"
}

adb_ shell true
adb_ shell "su -c 'mkdir -p $REMOTE/bin $REMOTE/models $REMOTE/runs && chmod 777 $REMOTE'"
if [ "${PUSH:-1}" = 1 ]; then
    for b in llama-ep-check llama-completion llama-cli test-backend-ops; do
        [ -f "$BUILD/bin/$b" ] && push_as_root "$BUILD/bin/$b" "$REMOTE/bin/$b" 755
    done
    push_as_root "$HERE/phone-cell.sh" "$REMOTE/phone-cell.sh" 755
fi
m=$(basename "$MODEL_HOST")
want=$(stat -c %s "$MODEL_HOST")
have=$(adb_ shell "stat -c %s $REMOTE/models/$m 2>/dev/null" | tr -d '\r' || true)
[ "$have" = "$want" ] || push_as_root "$MODEL_HOST" "$REMOTE/models/$m" 644
rmdir "$STAGE" 2> /dev/null || true

OUT=$REMOTE/runs/$NAME
wrapper=$(mktemp)
cat > "$wrapper" <<EOF
export BIN=$REMOTE/bin MODEL=$REMOTE/models/$m OUT=$OUT
rm -rf \$OUT; mkdir -p \$OUT
echo ep-$NAME > /sys/power/wake_lock
OUT=\$OUT CPUS='${CPUS:-f0}' PIN='${PIN:-}' SAMPLE_MS='${SAMPLE_MS:-100}' CMD='cd \$OUT && ${CMD//\'/\'\\\'\'}' sh $REMOTE/phone-cell.sh
echo ep-$NAME > /sys/power/wake_unlock
chmod -R a+rX \$OUT
EOF
push_as_root "$wrapper" "$REMOTE/cell.sh" 755
rm -f "$wrapper"
adb_ shell "su -c 'sh $REMOTE/cell.sh'"
mkdir -p "$RESULTS/$NAME"
for f in $(adb_ shell "ls $OUT" | tr -d '\r'); do
    adb_ exec-out "cat $OUT/$f" > "$RESULTS/$NAME/$f"
done
echo "exit=$(tr -d '\r\n' < "$RESULTS/$NAME/exit_code" 2>/dev/null) -> $RESULTS/$NAME"
