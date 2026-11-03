#!/usr/bin/env python3
# Summarize expert executor traces (LLAMA_EP_TRACE csv, and its .ops.csv from LLAMA_EP_OPPROF=1).
#   python trace-summary.py [--skip-tokens N] trace-a.csv [trace-b.csv ...]
# Phases are host wall time (CLOCK_MONOTONIC) per layer callback; medians over calls with n_tokens == 1
# after skipping the first N decode positions (warm-up). Branch windows: CPU = compute start..sync end on the
# main thread; GPU = submit start..sync end on the worker thread. Overlap is the intersection of the two windows,
# not proof that useful GEMVs ran at the same time.
import csv
import statistics as st
import sys
from collections import defaultdict


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def med(xs):
    return st.median(xs) if xs else float('nan')


def summarize(path, skip):
    calls = rows(path)
    for c in calls:
        for k, v in c.items():
            if k != 'assignment':
                c[k] = int(v)
    layers = max(c['layer'] for c in calls) + 1
    dec = [c for c in calls if c['n_tokens'] == 1]
    warm = dec[skip * layers:]
    ph = defaultdict(list)
    for c in warm:
        ph['total'].append(c['t_cleanup'] - c['t_enter'])
        ph['read+policy'].append(c['t_policy'] - c['t_enter'])
        ph['prep_cpu'].append(c['t_prep_cpu'] - c['t_policy'])
        ph['prep_gpu'].append(c['t_prep_gpu'] - c['t_prep_cpu'])
        if c['n_cpu']:
            ph['cpu_run'].append(c['t_cpu1'] - c['t_cpu0'])
        if c['n_gpu']:
            ph['gpu_submit'].append(c['t_gpu_submit1'] - c['t_gpu_submit0'])
            ph['gpu_run'].append(c['t_gpu_sync'] - c['t_gpu_submit0'])
        if c['n_cpu'] and c['n_gpu']:
            ov = min(c['t_cpu1'], c['t_gpu_sync']) - max(c['t_cpu0'], c['t_gpu_submit0'])
            ph['overlap'].append(max(ov, 0))
        ph['join..out'].append(c['t_out'] - c['t_join'])
        ph['cleanup'].append(c['t_cleanup'] - c['t_out'])
        ph['MB_cpu'].append(c['bytes_cpu'] / 1e6)
        ph['MB_gpu'].append(c['bytes_gpu'] / 1e6)
    n_cpu = sum(c['n_cpu'] for c in calls)
    n_gpu = sum(c['n_gpu'] for c in calls)
    print(f'== {path}: {len(calls)} calls ({len(warm)} warm decode), {layers} layers, slots cpu={n_cpu} gpu={n_gpu}')
    print('   ' + '  '.join(f'{k}={med(v)/1000:.3f}ms' if not k.startswith('MB') else f'{k}={med(v):.2f}' for k, v in ph.items()))
    ops_path = path + '.ops.csv'
    try:
        ops = rows(ops_path)
    except FileNotFoundError:
        return
    warm_inv = {c['invocation'] for c in warm}
    per = defaultdict(lambda: defaultdict(int))  # (backend, op) -> invocation -> us
    for o in ops:
        inv = int(o['invocation'])
        if inv in warm_inv:
            kind = o['op'] + ('.' + o['name'].rsplit('.', 1)[-1] if o['op'] in ('CPY', 'GET_ROWS') else '')
            per[('gpu' if o['gpu'] == '1' else 'cpu', kind)][inv] += int(o['t1']) - int(o['t0'])
    print(f'   per-op medians per call (serialized op-by-op, includes per-op launch/sync): {ops_path}')
    for (b, kind), d in sorted(per.items()):
        print(f'     {b} {kind:18s} {med(list(d.values()))/1000:8.3f} ms  ({len(d)} calls)')


def main():
    args = sys.argv[1:]
    skip = 2
    if args[:1] == ['--skip-tokens']:
        skip = int(args[1])
        args = args[2:]
    for p in args:
        summarize(p, skip)


if __name__ == '__main__':
    main()
