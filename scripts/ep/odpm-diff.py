#!/usr/bin/env python3
# Per-rail energy and mean power over one phone-cell run, from its ODPM snapshots (Pixel 7 / 7 Pro).
#   python odpm-diff.py RESULT_DIR [RESULT_DIR ...]
# energy_value counters are cumulative microjoules with a per-device timestamp T in ms. These are on-chip rails
# only (no display, storage or battery losses); the whole cell includes model load and every run in CMD.
import re
import sys
from pathlib import Path

LINE = re.compile(r'CH\d+\(T=(\d+)\)\[([^\]]+)\],\s*(\d+)')


def read(path):
    rails = {}
    for m in LINE.finditer(Path(path).read_text()):
        rails[m.group(2)] = (int(m.group(1)), int(m.group(3)))
    return rails


def main():
    for d in sys.argv[1:]:
        a, b = read(Path(d) / 'odpm-before.txt'), read(Path(d) / 'odpm-after.txt')
        print(f'== {d}')
        total = 0.0
        for name, (t1, e1) in b.items():
            t0, e0 = a[name]
            dt = (t1 - t0) / 1000
            mj = (e1 - e0) / 1000
            total += mj
            if mj / max(dt, 1e-9) >= 5:
                print(f'   {name:22s} {mj/1000:9.2f} J  {mj/max(dt, 1e-9):8.1f} mW')
        print(f'   {"all rails":22s} {total/1000:9.2f} J over {dt:.1f} s')


if __name__ == '__main__':
    main()
