#!/usr/bin/env python3
# Table of a policy x thread-setup sweep: decode ms/token (llama-completion "eval time") and per layer medians
# from the matching trace-<cfg>-<policy>.csv.
#   python compare-table.py RESULT_DIR
import re
import subprocess
import sys
from pathlib import Path

d = Path(sys.argv[1])
here = Path(__file__).resolve().parent
rows = []
cur = None
for line in (d / 'stdout.log').read_text().splitlines():
    if line.startswith('== '):
        cur = line[3:].split()
    elif cur and 'eval time' in line and 'prompt' not in line:
        ms = float(re.search(r'\(\s*([\d.]+) ms per token', line).group(1))
        rows.append((cur, ms))
print(f'{"setup":18s} {"policy":12s} {"ms/tok":>8s} {"layer":>7s} {"cpu":>7s} {"gpu":>7s} {"overlap":>8s}')
for cur, ms in rows:
    cfg, policy = cur[0], cur[-1]
    trace = d / f'trace-{cfg}-{policy}.csv'
    ph = {}
    if trace.exists():
        out = subprocess.run([sys.executable, str(here / 'trace-summary.py'), str(trace)], capture_output=True, text=True).stdout
        ph = dict(re.findall(r'(\w+)=([\d.]+)ms', out))
    f = lambda k: ph.get(k, '-')
    print(f'{" ".join(cur[:-1]):18s} {policy:12s} {ms:8.1f} {f("total"):>7s} {f("cpu_run"):>7s} {f("gpu_run"):>7s} {f("overlap"):>8s}')
