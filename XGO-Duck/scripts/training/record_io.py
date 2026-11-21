"""Bounded atomic JSON publication for records on a Windows-mounted drive.

A transient sharing/permission error must not discard a valid PPO update.
Retry only PermissionError for at most one second; retain the old complete
JSON until replacement succeeds. Persistent errors still stop the run.
"""
from pathlib import Path
import json
import time


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    deadline=time.monotonic()+1.
    while True:
        try:
            temporary.write_text(payload)
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.05)


def install():
    # Change only the new process's record writer. Frozen historical files and
    # experiment recipes remain byte-identical and available for audit.
    import training_watch
    import guarded_training
    training_watch.write=guarded_training.write=write
