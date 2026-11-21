"""CPU review of the explicit startup-only repair and unchanged probe budget."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
import recover_getup_value_startup as repair
import verify_getup_value_bootstrap as base


if __name__=='__main__':
    repair.activate();base.OUT=repair.OUT;base.PLAN=repair.PLAN
    p=repair.base.read(repair.PLAN);parent=repair.base.read(repair.PARENT_PLAN)
    for key,value in parent.items():
        if key not in ('source_sha256','library_source_sha256'):assert p[key]==value,key
    assert p['startup_recovery']['original_physics_steps']==0
    base.main()
