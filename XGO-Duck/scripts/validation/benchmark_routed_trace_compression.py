"""Compare two lossless NPZ settings on a captured seven-command trace chunk."""

from hashlib import sha256
import os
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write


def signature(arrays):
    return {name: dict(dtype=str(a.dtype), shape=list(a.shape),
        sha256=sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()) for name, a in arrays.items()}


def measure(arrays, level, expected):
    with NamedTemporaryFile(mode='w+b', suffix='.npz', dir=ROOT/'workspace') as temporary:
        start = perf_counter()
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=level) as archive:
            for name, a in arrays.items():
                with archive.open(f'{name}.npy', 'w', force_zip64=True) as member:
                    np.lib.format.write_array(member, a, allow_pickle=False)
        temporary.flush()
        os.fsync(temporary.fileno())
        seconds = perf_counter()-start
        size = os.fstat(temporary.fileno()).st_size
        temporary.seek(0)
        with np.load(temporary, allow_pickle=False) as saved:
            assert signature(saved) == expected
    return dict(level=level, elapsed_seconds=seconds, bytes=size, roundtrip_arrays=len(expected))


def main():
    plan = read(ROOT/'configs/experiments/continuous_routed_seven_command.json')
    record = ROOT/plan['record']
    state = read(record/'evaluation.json')
    assert state['status'] == 'completed'
    chunk, = [row for row in state['record_chunks'] if row['first'] == 601]
    source = record/chunk['file']
    assert sha(source) == chunk['sha256']
    with np.load(source, allow_pickle=False) as data:
        arrays = dict(data)
    expected = signature(arrays)
    rows = [measure(arrays, level, expected) for level in (6, 1, 1, 6)]
    out = ROOT/'records/reports/continuous-routed-seven-command/trace-compression-benchmark.json'
    assert not out.exists()
    write(out, dict(status='passed-lossless-npz-benchmark', physical_substeps=0, learning_updates=0,
        file=source.relative_to(ROOT).as_posix(), sha256=chunk['sha256'],
        original_bytes=source.stat().st_size, scratch_files_removed=True, results=rows))
    print('ROUTED_TRACE_COMPRESSION', rows, flush=True)


if __name__ == '__main__':
    main()
