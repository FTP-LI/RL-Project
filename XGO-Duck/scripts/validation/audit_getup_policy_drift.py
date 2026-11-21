"""CPU counterfactual action audit on fixed official getup observations.

Small per-update PPO KL need not limit cumulative displacement from the
official policy. Evaluate both closed trained actors on the SAME official
inputs, using prospectively fixed times and all320worlds. Original success
labels only stratify saved inputs; they do not select treatment outcomes.
The fixed-sigma displacement is a descriptive proxy, not an official policy
distribution, causal recovery effect or new simulation/learning result.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/getup-policy-drift'
DEFINITION=ROOT/'configs/experiments/getup_policy_drift_audit.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(),'Retain the first saved-data audit and its definition'
    d=read(DEFINITION);basis=ROOT/d['basis'];proof=read(basis)
    assert proof['status']=='passed'
    source=ROOT/proof['record']/'coverage-trace.npz';assert sha(source)==proof['trace_sha256']
    times=np.asarray(d['input_state_times_seconds']);indices=np.rint(times/d['step_dt']).astype(int)
    with np.load(source) as f:
        assert f['observations'].shape==(550,320,61)
        np.testing.assert_allclose(f['time'],np.arange(1,551)*d['step_dt'],atol=1e-7,rtol=0)
        inputs=f['observations'][indices].copy()
        recorded=f['getup_actions'][indices].copy()
    end=np.asarray(proof['end_hold']['stable_by_environment'],bool)
    assert len(end)==320 and int(end.sum())==295
    models={};predictions={};providers={}
    for role in ['original',*d['roles']]:
        evidence=read(ROOT/f'records/reports/getup-late-feasibility/{role}-analysis.json')
        p=read(ROOT/evidence['record']/'protocol.json');model=ROOT/p['model']
        assert sha(model)==p['model_sha256']
        options=ort.SessionOptions();options.intra_op_num_threads=1
        session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
        assert session.get_providers()==['CPUExecutionProvider']
        flat=inputs.reshape(-1,61)
        predictions[role]=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in flat]).reshape(len(times),320,14)
        models[role]=dict(path=p['model'],sha256=p['model_sha256']);providers[role]=session.get_providers()
    np.testing.assert_allclose(predictions['original'],recorded,atol=2e-5,rtol=2e-5)
    rows=[];groups={'all':np.ones(320,bool),'original-held':end,'original-failed':~end}
    sigma=d['fixed_sigma_rad'];proxies={}
    for role in d['roles']:
        delta=predictions[role]-predictions['original']
        proxy=np.square(delta).sum(-1)/(2*sigma*sigma);proxies[role]=proxy
        for k,t in enumerate(times):
            for name,mask in groups.items():
                value=delta[k,mask]
                rows.append(dict(role=role,input_state_seconds=float(t),group=name,worlds=int(mask.sum()),
                    action_rms_rad=float(np.sqrt(np.square(value).mean())),
                    action_absolute_max_rad=float(abs(value).max()),
                    displacement_mean=float(proxy[k,mask].mean()),displacement_median=float(np.median(proxy[k,mask])),
                    displacement_p90=float(np.quantile(proxy[k,mask],.9))))
    logs={}
    for role in d['roles']:
        # Evaluation run IDs belong to seed1016. The exported artifact retains
        # the actual seed42 training run ID; use that provenance for its log.
        folder=ROOT/'records/experiments'/Path(models[role]['path']).parent.name
        file=folder/'learning-metrics.jsonl';metrics=[json.loads(x) for x in file.read_text().splitlines() if x.strip()]
        assert len(metrics)==50 and metrics[-1]['completed_updates']==50
        logs[role]=dict(path=str(file.relative_to(ROOT)),sha256=sha(file),updates=50,
            recorded_per_update_kl_max=max(x['kl'] for x in metrics),last_per_update_kl=metrics[-1]['kl'],
            kl_guard_triggered_updates=sum(bool(x['actor_kl_limited']) for x in metrics),
            stored_probe_action_drift_rms=metrics[-1]['raw_action_drift_rms'],
            stored_probe_action_drift_max=metrics[-1]['raw_action_drift_max'])
    OUT.mkdir();archive=OUT/'fixed-input-comparison.npz'
    np.savez_compressed(archive,inputs=inputs,input_state_times=times,original_end_hold=end,
        **{f'{k}_mean':v for k,v in predictions.items()},**{f'{k}_displacement':v for k,v in proxies.items()})
    result=dict(status='passed-saved-data-only',definition_sha256=sha(DEFINITION),basis_sha256=sha(basis),
        training_pair_sha256=sha(ROOT/d['training_pair']),
        source_trace_sha256=sha(source),script_sha256=sha(Path(__file__)),models=models,providers=providers,
        observations_each_model=int(inputs.shape[0]*320),original_held=295,original_failed=25,
        reference_reconstruction_max_error=float(abs(predictions['original']-recorded).max()),
        fixed_sigma_rad=sigma,rows=rows,training_logs=logs,archive_sha256=sha(archive),
        physics_steps=0,optimizer_steps=0,heldout_used=False,scope=d['drift_proxy'])
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('GETUP_DRIFT_AUDIT',json.dumps({k:v for k,v in result.items() if k not in ('rows','models','providers')}),flush=True)
    for row in rows:
        if row['input_state_seconds'] in (3.,7.98,10.98) and row['group']=='all':print('DRIFT_AT_TIME',row,flush=True)


if __name__=='__main__':main()
