"""Compare cached and endpoint progress reward on the same recorded movement."""

from pathlib import Path
import re

import numpy as np
import torch

from record_io import write
from review_roll_forward_input_stop import ROOT, arrays, read, sha
from roll_endpoint_progress import advance_progress
from roll_metrics import progress
from roll_stand_walk_sequence import ROLL, STAND_LANDING


REPORT = ROOT/'records/reports/roll-forward-input-integration/support-window-force'


def verify_boundaries(knobs):
    z = torch.zeros(2,dtype=torch.float64)
    contact = torch.tensor([False,True])
    onset = torch.tensor([.25,.25],dtype=torch.float64)
    _,s,f,pay = advance_progress(z,z,z,onset,z,contact,**knobs)
    torch.testing.assert_close(s,torch.tensor([0.,.25],dtype=torch.float64))
    torch.testing.assert_close(pay,torch.tensor([0.,knobs['max_paid_rate']*knobs['step_dt']/knobs['target_angle']],dtype=torch.float64))
    classed = torch.ones(2,dtype=torch.bool)
    g2,s2,f2,back = advance_progress(onset,s,f,-onset,z,classed,**knobs)
    assert not back.any() and not g2.any() and s2[0]<0.
    _,_,again,still = advance_progress(g2,s2,f2,onset,z,classed,**knobs)
    assert not still.any()
    torch.testing.assert_close(again,f)
    full = z+knobs['target_angle']
    _,_,_,completed = advance_progress(full,full,full,onset,z,classed,**knobs)
    assert not completed.any()


def main():
    torch.set_num_threads(1)
    project = read(ROOT/'project.json')
    prior = project['goal_execution']['roll_friction_paired']
    training = ROOT/prior['protocol']
    assert sha(training)==prior['sha256']
    arm = read(training)['arms']['low-friction']
    configuration = ROOT/'records/experiments'/arm['run_id']/'environment.yaml'
    source = configuration.read_text(encoding='utf-8')
    term = re.search(r'^  roulade_progress:\n(.*?)(?=^  \w+:)',source,re.MULTILINE | re.DOTALL)[1]
    assert 'mjlab_microduck.tasks.mdp.roulade_progress' in term and '\nscale_rewards_by_dt: true\n' in source
    values = {key:float(re.search(r'^\s+'+key+r': ([0-9.]+)$',term,re.MULTILINE)[1]) for key in ('weight','target_angle','max_paid_rate')}
    audit_path, proof_path = (REPORT/name for name in ('reward-contact-timing-audit.json','force-readback.json'))
    audit, proof = read(audit_path),read(proof_path)
    assert audit['run_id']==proof['run_id'] and sha(proof_path)==project['goal_execution']['roll_support_window_probe']['result_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in audit['source_sha256'].items())
    assert all(sha(ROOT/name)==digest for name,digest in proof['source_sha256'].items())
    execution = audit['installed_reward_order']
    env = Path(execution['env'])
    manager_path = env.parent.parent/'managers/reward_manager.py'
    lines = manager_path.read_text(encoding='utf-8').splitlines()
    scale = next(i for i,line in enumerate(lines,1) if 'scale = dt if self._scale_by_dt else 1.0' in line)
    unscaled = next(i for i,line in enumerate(lines,1) if 'self._step_reward[:, term_idx] = value / scale' in line)
    root = ROOT/'records/evaluations'/proof['run_id']
    plan, trace = read(root/'protocol.json'),arrays(root/'trace.npz')
    assert sha(root/'protocol.json')==proof['protocol_sha256']
    endpoint = root/'endpoint-contact.npz'
    assert sha(endpoint)==audit['source_sha256'][endpoint.relative_to(ROOT).as_posix()]
    cache = arrays(endpoint)
    start = plan['initial_stand_controls']
    n = plan['worlds']
    knobs = dict(step_dt=plan['roll']['criteria']['step_dt'],target_angle=values['target_angle'],max_paid_rate=values['max_paid_rate'])
    verify_boundaries(knobs)
    angle,_,matrix = progress({key:trace[key][start:] for key in ('quaternion','ground_contact')})
    inc = torch.as_tensor(np.diff(angle,axis=0),dtype=torch.float64)
    lateral = torch.as_tensor(matrix[1:,:,2,1],dtype=torch.float64)
    rolling = torch.as_tensor(trace['active'][start:-1] & (trace['phase_next'][start:-1]==ROLL))
    pay = {}
    for sample in ('cached','fresh'):
        signal = torch.as_tensor(cache[sample][start+1:,:,0])
        geom = supported = frontier = torch.zeros(n,dtype=torch.float64)
        payments = []
        for displacement,axis,contact,active in zip(inc,lateral,signal,rolling,strict=True):
            geom,supported,frontier,amount = advance_progress(geom,supported,frontier,displacement*active,
                                                               axis,contact,**knobs)
            payments.append(amount*values['weight'])
        pay[sample] = torch.stack(payments).numpy()
        assert not pay[sample][~rolling.numpy()].any()
    accepted = (trace['phase_next'][:]==STAND_LANDING).any(axis=0)
    assert int(accepted.sum())==proof['original_roll_success']
    cap = knobs['max_paid_rate']*knobs['step_dt']/knobs['target_angle']*values['weight']
    difference = pay['fresh']-pay['cached']
    assert np.abs(difference).max()<=cap+1e-10
    summarize = {}
    for label,mask in (('roll-pass',accepted),('roll-fail',~accepted)):
        sum_old,sum_new = (pay[role][:,mask].sum(axis=0) for role in ('cached','fresh'))
        delta = sum_new-sum_old
        summarize[label] = dict(worlds=int(mask.sum()),median_cached_payment=float(np.median(sum_old)),
            median_fresh_payment=float(np.median(sum_new)),paired_median_payment_change=float(np.median(delta)),
            worlds_with_lower_payment=int((delta<0.).sum()),worlds_with_higher_payment=int((delta>0.).sum()))
    result = REPORT/'endpoint-progress-payment-replay.json'
    write(result,dict(run_id=proof['run_id'],reward_settings=dict(values,step_dt=knobs['step_dt']),
        scope='Same recorded integrated poses, with either of the two recorded contact readings',
        checked_boundaries=['unsupported-progress','rate-cap','reversal','repeated-frontier','one-revolution-cap'],
        archived_scene_group=summarize,max_abs_step_difference=float(np.abs(difference).max()),
        matched_reward_payment_units=dict(scale_line=scale,step_reward_line=unscaled,source=str(manager_path),source_sha256=sha(manager_path)),
        source_sha256={item.relative_to(ROOT).as_posix():sha(item) for item in (training,configuration,audit_path,proof_path,endpoint,
            Path(__file__),ROOT/'scripts/training/roll_endpoint_progress.py')},new_physical_steps=0,learning_updates=0,hardware_io=False))
    print('ENDPOINT_PROGRESS_REPLAY',summarize,flush=True)


if __name__=='__main__':
    main()
