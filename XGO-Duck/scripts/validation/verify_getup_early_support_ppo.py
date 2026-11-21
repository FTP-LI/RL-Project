"""Independently reload both completed PPO runs before functional evaluation."""
import torch

from run_getup_early_support_ppo import OUT, PLAN, ROLES, arm_plan, checked
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_early_support_rollout import readback


def main():
    torch.set_num_threads(4)
    spec = checked()
    coordinator = read(OUT/'coordinator.json')
    assert coordinator['status']=='collected-review-required' and coordinator['protocol_sha256']==sha(PLAN)
    assert all(coordinator['arms'][role]['exit_code']==0 for role in ROLES)
    warm = torch.load(ROOT/spec['critic_checkpoint'],map_location='cpu',weights_only=True)
    arms = {}
    for role in ROLES:
        p = arm_plan(spec,role)
        record = ROOT/'records/experiments'/p['run_id']
        arms[role] = readback(spec,p,record,warm)
        write(OUT/f'{role}-verification.json',arms[role])
    watch = read(OUT/'windows-observations.json')
    assert watch['status']=='passed'
    for role, result in arms.items():
        watched = {row['latest_clip']['sha256'] for row in watch['observations']
                   if row['role']==role and 'latest_clip' in row}
        assert watched and watched.issubset({clip['sha256'] for clip in result['videos']})
    assert not torch.cuda.is_initialized()
    result = dict(status='passed-online-training-integrity',protocol_sha256=sha(PLAN),arms=arms,
                  windows_observation_sha256=sha(OUT/'windows-observations.json'),
                  functional_evaluation_pending=True,heldout_used=False)
    write(OUT/'training-verification.json',result)
    print('EARLY_SUPPORT_PPO_VERIFIED',{role: item['actual_training_transitions'] for role,item in arms.items()},flush=True)


if __name__=='__main__':
    main()
