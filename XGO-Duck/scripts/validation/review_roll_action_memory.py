"""Replay runtime-style output memory on saved raw roll requests, CPU only.

This quantifies different requested targets, not changed physical motion.
It preserves the old 28/32 verdict and cannot predict candidate successes.
"""
from pathlib import Path
import json
import numpy as np
from roll_action_memory import ActionMemory
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def main():
    memory=ActionMemory(2);one=np.ones((2,14),np.float32)
    np.testing.assert_allclose(memory.update(one),.85,atol=1e-7)
    np.testing.assert_allclose(memory.update(one),.9775,atol=1e-7)
    memory.reset([0]);a=memory.update(one)
    np.testing.assert_allclose(a[0],.85,atol=1e-7);assert np.all(a[1]>.99)
    memory.reset([0,1]);np.testing.assert_array_equal(memory.value,0.)
    bad=one.copy();bad[0,0]=np.nan
    try:memory.update(bad)
    except AssertionError:pass
    else:raise AssertionError('Invalid actions accepted')
    p=read(ROOT/'records/reports/roll-reference/progress.json');folder=Path(p['path']);report=read(folder/'evaluation.json')
    item=report['conditions'][0];path=folder/'forward-roll-trace.npz';assert sha(path)==item['trace_sha256']
    with np.load(path) as t:
        raw=t['raw_policy_actions'].copy();active=t['active'].copy()
    assert raw.shape==(451,32,14);memory=ActionMemory(32);filtered=np.zeros_like(raw)
    for step in range(1,len(raw)):filtered[step]=memory.update(raw[step])
    delta_raw=np.diff(raw,axis=0);delta_filtered=np.diff(filtered,axis=0);valid=active[1:]
    runtime=ROOT/'official_resources/xgoduck_runtime_arduino/python/policy.py'
    text=runtime.read_text();assert 'self._act_filt[sl] *= alpha' in text and 'action[sl] * (1 - alpha)' in text
    result=dict(status='request-replay-only',source_successes=item['successes'],source_episodes=32,
        raw_delta_rms=float(np.sqrt(np.square(delta_raw[valid]).mean())),
        filtered_delta_rms=float(np.sqrt(np.square(delta_filtered[valid]).mean())),
        maximum_requested_difference_deg=float(np.rad2deg(abs(filtered[active]-raw[active]).max())),
        alpha_old_value=.15,raw_new_value=.85,source_trace_sha256=sha(path),runtime_source_sha256=sha(runtime),
        implementation_sha256=sha(ROOT/'scripts/validation/roll_action_memory.py'),script_sha256=sha(Path(__file__)),
        checks=dict(numeric_recurrence=True,selective_reset=True,nonfinite_rejected=True),
        environment_steps=0,learning_updates=0,original_verdict_unchanged=True,
        scope='Hypothetical requests on original raw sequence; no candidate dynamics or success estimate')
    out=ROOT/'records/reports/roll-action-memory';out.mkdir(exist_ok=True);write(out/'prior-action-review.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
