"""Reconstruct and restore the controller at the first forward decision."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from getup_release_fork_state import digest, signature
from record_io import write
from roll_stand_walk_sequence import FORWARD, STATE_FIELDS, Sequence


OUT = ROOT/'records/reports/continuous-roll-projected-entry'
DRAFT = ROOT/'configs/experiments/continuous_forward_yaw_pair_draft.json'
PHASE = ('phase','dwell','stand_at','forward_at','phase_changed')
ROLL = ('pitch','angle','supported','seen_feet','seen_head','finished','intact','dwell','first_ready','step')


def controller_snapshot(sequence):
    return {f'{prefix}.{name}':np.asarray(getattr(owner,name)).copy()
            for prefix,owner,names in (('phase',sequence,PHASE),('roll',sequence.roll_latch,ROLL))
            for name in names}


def restore_controller(sequence,record):
    for key,value in record.items():
        prefix,name = key.split('.')
        owner = sequence if prefix=='phase' else sequence.roll_latch
        current = getattr(owner,name)
        if isinstance(current,np.ndarray):
            np.copyto(current,value,casting='no')
        else:
            assert value.ndim==0
            setattr(owner,name,value.item())


def main():
    p = read(DRAFT)
    evidence = ROOT/p['parent_result']
    proof = read(evidence)
    assert proof['status']=='passed-independent-verification' and not proof['heldout_used']
    assert not proof['passed_development_gate'] and proof['counts']['entered_forward']==126
    parent = ROOT/p['parent_protocol']
    assert sha(parent)==proof['protocol_sha256']
    record = ROOT/proof['record']
    trace_path,cmd_path = record/'trace.npz',record/'commands.npz'
    for path in (trace_path,cmd_path):
        assert sha(path)==proof['source_sha256'][path.relative_to(ROOT).as_posix()]
    design = read(parent)
    first = min(k for k in proof['forward_at'] if k>=0)
    seq = Sequence(design)
    with np.load(trace_path) as data:
        trace = {name:data[name][:first+3] for name in (*STATE_FIELDS,'phase_next')}
    for k in range(first+1):
        seq.update(k,{name:trace[name][k] for name in STATE_FIELDS})
        np.testing.assert_array_equal(seq.phase,trace['phase_next'][k])
    selected = np.flatnonzero(seq.phase==FORWARD)
    assert selected.size and np.array_equal(selected,np.flatnonzero(np.asarray(proof['forward_at'])==first))
    frozen = controller_snapshot(seq)
    h = digest(signature(frozen))
    next_state = {name:trace[name][first+1] for name in STATE_FIELDS}
    seq.update(first+1,next_state)
    expected_next = controller_snapshot(seq)
    assert digest(signature(expected_next))!=h
    restore_controller(seq,frozen)
    assert digest(signature(controller_snapshot(seq)))==h
    seq.update(first+1,next_state)
    assert digest(signature(controller_snapshot(seq)))==digest(signature(expected_next))
    with np.load(cmd_path) as commands:
        observed = commands['observed'][first-1]
        hip = [20+design['walking']['joint_names'].index(f'{side}_hip_pitch') for side in ('left','right')]
        memory = dict(previous_features=np.column_stack((observed[:,2],observed[:,hip].sum(axis=1))),
                      history=commands['raw_policy_actions'][first-1],previous=commands['applied_targets'][first-1])
        np.testing.assert_array_equal(commands['external_command'][:first,:,0],0.)
        np.testing.assert_array_equal(commands['external_command'][first,selected,0],np.float32(.15))
    output = OUT/'forward-yaw-fork-controller-preflight.json'
    state = OUT/'forward-yaw-fork-controller.npz'
    assert not output.exists() and not state.exists()
    np.savez_compressed(state,**frozen,**memory)
    write(output,dict(status='passed-offline-forward-controller-restore',first_forward_control=first,
        earliest_worlds=selected.tolist(),controller_fields=len(frozen),policy_memory_fields=list(memory),
        controller_state_sha256=h,archive_sha256=sha(state),replayed_next_boundary_exact=True,
        sources={path.relative_to(ROOT).as_posix():sha(path) for path in (DRAFT,parent,evidence,trace_path,cmd_path,Path(__file__))},
        physical_substeps=0,learning_updates=0,heldout_used=False))
    print('CONTINUOUS_FORWARD_FORK',first,len(selected),len(frozen),h,flush=True)


if __name__=='__main__':
    main()
