"""Reproduce the saved reward and exercise a partial reset on the real component."""
from pathlib import Path
import json
import sys

import numpy as np
import torch

from check_getup_right_side_upright_progress import DEST as SCREEN, PLAN, potential
from review_getup_right_side_trigger_transfer import COHORTS, DEST as TRIGGER, ROOT, load, sha
from standup_metrics import projected_gravity

sys.path.insert(0,str(ROOT/'scripts/training'))
from getup_right_upright_progress import RightUprightProgress


DEST = SCREEN.with_name('right-side-upright-reward-cpu.json')


def reset_fixture(entry,horizon):
    device = 'cpu'
    state = RightUprightProgress(3,device,100,.75,entry,horizon)
    observed = torch.zeros((3,61))
    observed[:,3:6] = torch.tensor([[0.,-1.,0.],[0.,-1.,0.],[-1.,0.,0.]])
    state.observe_release(observed,torch.full((3,),100))
    on = torch.ones((3,2),dtype=torch.bool)
    contact = torch.ones(3,dtype=torch.bool)
    gravity = torch.full((3,),-np.cos(np.deg2rad(32.)),dtype=torch.float32)
    for k in range(101,106):
        assert not state.advance(gravity,on,contact,torch.full((3,),k),contact).any()
    assert state.trigger_age.tolist()==[105,105,-1]
    state.reset(torch.tensor([0]))
    assert state.trigger_age.tolist()==[-1,105,-1] and state.right.tolist()==[False,True,False]
    zero = state.advance(torch.full((3,),-.99),on,contact,torch.tensor([1,106,106]),contact)
    assert zero[0]==zero[2]==0. and zero[1]>0.
    # Only the non-reset world's original reward state persists.
    state.observe_release(observed,torch.tensor([100,106,106]))
    for k in range(101,105):
        state.advance(gravity,on,contact,torch.tensor([k,106+k,106+k]),contact)
        assert state.trigger_age[0]==-1
    state.advance(gravity,on,contact,torch.tensor([105,211,211]),contact)
    assert state.trigger_age.tolist()==[105,105,-1]


def main():
    assert not DEST.exists() and not torch.cuda.is_initialized()
    draft,screen,evidence = load(PLAN),load(SCREEN),load(TRIGGER)
    assert screen['passing_examples_receive_nonnegative_progress'] and screen['untriggered_and_other_pose_signal_exactly_zero']
    for receipt in (screen,evidence):
        for path,digest in receipt['source_sha256'].items():
            assert sha(ROOT/path)==digest,path
    entry = evidence['reused_entry']
    horizon = draft['proposed_reward_window_controls']
    torch.set_num_threads(1)
    reset_fixture(entry,horizon)
    rows = {}
    for name,proof_file,roles in COHORTS:
        proof = load(proof_file)
        record = ROOT/proof['record']
        names = ('quaternion','foot_contact','native_head_leg_contact','active','actor_input')
        with np.load(record/'trace-shared-prefix.npz') as file:
            prefix = {key:file[key] for key in names}
        local = {}
        for role in roles:
            with np.load(record/f'trace-{role}.npz') as file:
                data = {key:np.concatenate((prefix[key],file[key]),axis=0) for key in names}
            n = data['quaternion'].shape[1]
            ages = torch.arange(1,len(data['active'])+1,dtype=torch.long)[:,None].expand(-1,n)
            state = RightUprightProgress(n,'cpu',100,.75,entry,horizon)
            released_actor = torch.from_numpy(data['actor_input'][100])
            state.observe_release(released_actor,ages[99])
            labels = np.asarray([row['release_pose'] for row in proof['arms'][role]['timing']['episodes']])
            np.testing.assert_array_equal(state.right.numpy(),labels=='right-side')
            gravity = torch.from_numpy(projected_gravity(data['quaternion'])[...,2].astype(np.float32))
            feet,head,active = (torch.from_numpy(data[key]) for key in names[1:4])
            values = np.zeros((len(ages),n),dtype=np.float32)
            for i in range(100,round(entry['last_trigger_seconds']/.02)+horizon+1):
                values[i] = state.advance(gravity[i],feet[i],head[i],ages[i],active[i]).numpy()
            fired = state.trigger_age.numpy()
            expected = evidence['cohorts'][name]['references'][role]['trigger_control_by_world']
            assert {str(w):int(fired[w]) for w in np.flatnonzero(fired>=0)}==expected
            p = potential(data['quaternion'],data['foot_contact'])
            want = np.zeros_like(p)
            for key,k in expected.items():
                w = int(key)
                want[k:k+horizon,w] = np.diff(p[k-1:k+horizon,w])
            np.testing.assert_allclose(values,want,atol=6e-7,rtol=0)
            np.testing.assert_array_equal(values[:,fired<0],0.)
            local[role] = dict(right_release_latch_matches=True,trigger_count=len(expected),
                largest_absolute_reward_error=float(np.abs(values-want).max()),controls_checked=len(ages))
        rows[name] = local
    assert not torch.cuda.is_initialized()
    module = ROOT/'scripts/training/getup_right_upright_progress.py'
    receipt = dict(status='passed-cpu-reward-state-replay',physical_substeps=0,learning_updates=0,
        holdout_used=False,cases=rows,partial_reset_fixture=True,expected_events=28,
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (PLAN,SCREEN,TRIGGER,Path(__file__),module)},
        scope='CPU component readback; native CUDA clock, solver contact timing, value and PPO remain separate gates.')
    DEST.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('RIGHT_UPRIGHT_REWARD_CPU',rows,flush=True)


if __name__=='__main__':
    main()
