#!/usr/bin/env python3
# Keep the first N transformer blocks of a GGUF (for fast host-side tests of the expert executor).
# Logits of the result are meaningless; per-expert and cross-backend checks stay valid.
#   python truncate-gguf.py IN.gguf OUT.gguf N_LAYERS
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gguf-py'))
import gguf  # noqa: E402

src, dst, n_keep = sys.argv[1], sys.argv[2], int(sys.argv[3])
reader = gguf.GGUFReader(src)
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
block_key = f'{arch}.block_count'
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    value = field.contents()
    if field.name == block_key:
        assert value >= n_keep
        value = n_keep
    elif val_type == gguf.GGUFValueType.ARRAY and isinstance(value, list) and field.name.startswith(arch + '.') and len(value) == reader.fields[block_key].contents():
        value = value[:n_keep]  # per-layer arrays
    writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
keep = [t for t in reader.tensors if not (m := re.match(r'blk\.(\d+)\.', t.name)) or int(m.group(1)) < n_keep]
for t in keep:
    writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for t in keep:
    writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
writer.close()
print(f'{arch}: kept {len(keep)} of {len(reader.tensors)} tensors, {n_keep} blocks -> {dst}')
