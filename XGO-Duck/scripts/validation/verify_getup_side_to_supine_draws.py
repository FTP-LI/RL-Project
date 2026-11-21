"""Reconstruct committed replay draws and count the actual new coverage."""
from pathlib import Path

import numpy as np
import torch

from s1_reference_inputs import ROOT,read,sha
from getup_recovery_retention_ppo import retention_indices
from record_io import write
from run_getup_recovery_retention import verify_retention
from run_getup_side_to_supine_pair import OUT,PLAN,ROLES


def main():
    spec = read(PLAN)
    result = read(OUT/'training-verification.json')
    assert result['status']=='passed-online-training-integrity' and result['protocol_sha256']==sha(PLAN)
    p = spec['environment']
    cells = read(ROOT/p['retention_cells'])
    proofs, masks = {}, {}
    for role in ROLES:
        arm = result['arms'][role]
        folder = ROOT/arm['record']
        arm_plan = read(folder/'protocol.json')
        replay = ROOT/spec['replay_paths'][role]
        assert ROOT/arm_plan['retention_data']==replay and arm_plan['early_support_enabled']
        loaded = read(folder/'replay-start.json')
        assert loaded['status']=='passed-live-replay' and loaded['data_sha256']==sha(replay) and loaded['rows']==900
        assert loaded['early_support_enabled'] and loaded['physical_substeps']==0
        with np.load(replay) as data:
            assert np.isin(data['seed'],[1061,1062]).all()
            masks[role] = (data['actual_pose']=='supine')&np.isin(data['requested_pose'],('left-side','right-side'))
        proofs[role] = verify_retention(arm_plan,arm)
    assert proofs['control']['sample_hashes']==proofs['candidate']['sample_hashes']
    steps = proofs['control']['retention_batches']
    assert steps==proofs['candidate']['retention_batches']==2040
    assert not masks['control'].any() and masks['candidate'].sum()==180
    recorded = np.stack([retention_indices(cells,p['retention_seed'],i,p['retention_samples_per_cell']).numpy()
                         for i in range(steps)])
    selected = {role:int(mask[recorded].sum()) for role,mask in masks.items()}
    assert selected['candidate']>0 and selected['control']==0
    write(OUT/'replay-draws.json',dict(status='passed-identical-index-draws',protocol_sha256=sha(PLAN),
          training_verification_sha256=sha(OUT/'training-verification.json'),script_sha256=sha(Path(__file__)),
          side_to_supine_draws=selected,rows_drawn_per_arm=int(recorded.size),batches=steps,proofs=proofs,
          heldout_used=False,physical_substeps=0,learning_updates=0))
    assert not torch.cuda.is_initialized()
    print('SIDE_TO_SUPINE_DRAWS',steps,recorded.size,selected,flush=True)


if __name__=='__main__':
    main()
