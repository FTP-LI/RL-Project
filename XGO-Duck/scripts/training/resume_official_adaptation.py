"""Resume the recorded I/O interruption within its remaining compute budget.

No update is silently replayed. Completed but unsaved work is charged to the
original budget; resume only from the independently verified durable model.
"""
from pathlib import Path
import official_continuation as entry
from record_io import install,write

install()
entry.write=write
entry.PLAN=entry.ROOT/'configs/experiments/official_adaptation_recovery.json'
original_name=entry.run_name
def recovery_name(skill,change,arm,seed,stamp):
    return original_name(skill,'official-adaptation-recovery',arm,seed,stamp)
entry.run_name=recovery_name

if __name__=='__main__':entry.main()
