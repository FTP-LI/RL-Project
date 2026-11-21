"""Audit the two continuations and present cumulative progress with the pair budget."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from run_walk_backward_stall_capture import OUT, PLAN, ROLES, page
import training_live_page as live


def main():
    proof = OUT/'analysis.json'
    result = read(proof)
    assert result['status']=='passed-stall-capture-integrity' and result['protocol_sha256']==sha(PLAN)
    assert not result['learning_updates'] and not result['heldout_used']
    target = OUT/'continuation-audit.json'
    assert not target.exists()
    record = ROOT/result['record']
    paths = {role:record/f'trace-{role}.npz' for role in ROLES}
    assert all(sha(path)==result['trace_sha256'][role] for role,path in paths.items())
    spec = read(PLAN)
    k = spec['phase']['warmup_controls']
    selected = np.asarray(result['eligible'],dtype=np.int64)
    tol = spec['capture']['gate']['maximum_first_continued_qpos_difference']
    fields = ('qpos','observations','actions')
    differences = {}
    with np.load(paths[ROLES[0]]) as original,np.load(paths[ROLES[1]]) as restored:
        qpos = abs(original['qpos'][0]-restored['qpos'][0])
        active = original['active'][0]&restored['active'][0]
        assert qpos.shape[0]==len(active)
        limits = qpos.max(axis=1)
        above = np.flatnonzero(limits>tol)
        joint_names = dict(zip(read(record/'evaluation.json')['qpos_joint_addresses'],
                               spec['environment']['joint_names'],strict=True))
        first_qpos = dict(
            worlds_over_threshold=len(above),active_worlds_over_threshold=int(active[above].sum()),
            selected_worlds_over_threshold=np.intersect1d(selected,above).tolist(),
            first_contact_flags_differ=int(np.any(original['contact'][0]!=restored['contact'][0],axis=1).sum()),
            maximum_selected=float(limits[selected].max()) if len(selected) else 0.,
            examples=[dict(world=int(world),coordinate=int(qpos[world].argmax()),
                           joint=joint_names.get(int(qpos[world].argmax())),
                           maximum=float(limits[world]),active=bool(active[world]))
                      for world in above[np.argsort(limits[above])[-5:][::-1]]])
        for group,ids in (('all',slice(None)),('selected',selected)):
            differences[group] = {}
            for field in fields:
                a,b = original[field][:,ids],restored[field][:,ids]
                diff = np.max(abs(a-b),axis=(1,2)) if a.shape[1] else np.zeros(a.shape[0])
                visible = np.flatnonzero(diff>tol)
                differences[group][field] = dict(maximum=float(diff.max()),first=float(diff[0]),
                    first_control_above_display_threshold=int(k+1+visible[0]) if len(visible) else None)
    state_path = record/'state.json'
    state = read(state_path)
    runtime_state_sha = sha(state_path)
    windows = read(record/'windows-view-verification.json')
    assert windows['status']=='passed' and windows['observer_host']=='Windows'
    assert state['status']=='completed' and state['transitions']==result['transitions']
    budget = spec['maximum_control_transitions']//spec['environment']['parallel_worlds']
    assert budget==state['controls'] and state['maximum_controls']==spec['environment']['maximum_control_steps']
    # Keep the original publisher's state for the running observer receipt.
    # The final HTML uses the cumulative two-branch denominator.
    display = dict(state,maximum_controls=budget)
    display_path = record/'verified-display-state.json'
    assert not display_path.exists()
    with patch.object(live,'PAGE',page(result['seed'])),patch.object(live,'write',lambda _,value:write(display_path,value)):
        live.publish(record,display)
    assert sha(state_path)==runtime_state_sha
    write(target,dict(status='passed-continuation-and-final-view-audit',analysis_sha256=sha(proof),
        threshold=tol,differences=differences,first_qpos=first_qpos,
        windows_receipt_sha256=sha(record/'windows-view-verification.json'),
        runtime_state_sha256=runtime_state_sha,display_state_sha256=sha(display_path),
        displayed_cumulative_controls=budget,physical_substeps=0,learning_updates=0))
    print('BACKWARD_STALL_CONTINUATION_AUDIT',differences,first_qpos,'display',budget,flush=True)


if __name__=='__main__':
    main()
