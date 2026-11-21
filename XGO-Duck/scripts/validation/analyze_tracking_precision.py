"""Judge the frozen reward-width pair from independently verified traces."""
from copy import deepcopy
from pathlib import Path
import json
import yaml
from s1_reference_inputs import ROOT, read, sha
from analyze_official_bootstrap import metrics
from record_io import write

OUT=ROOT/'records/reports/tracking-precision'


def main():
    pair_path=ROOT/'configs/experiments/tracking_precision_pair.json';pair=read(pair_path)
    pre=read(OUT/'preflight.json');assert pre['protocol_sha256']==sha(pair_path)
    paths={'parent':Path(read(ROOT/'records/reports/official-actor-bootstrap/evaluation-progress.json')['evaluation'])}
    train_cfgs={}
    for arm in ('control','candidate'):
        progress=read(OUT/arm/'evaluation-progress.json');assert progress['status']=='collected-and-verified'
        paths[arm]=Path(progress['evaluation'])
        record=ROOT/'records/experiments'/pre['runs'][arm]
        params=Path(read(record/'runner.json')['log_dir'])/'params/env.yaml'
        train_cfgs[arm]=yaml.load(params.read_text(),Loader=yaml.BaseLoader)
        assert float(train_cfgs[arm]['rewards']['track_linear_velocity']['params']['std'])==(.25 if arm=='control' else .1)
    left,right=deepcopy(train_cfgs['control']),deepcopy(train_cfgs['candidate'])
    right['rewards']['track_linear_velocity']['params']['std']=left['rewards']['track_linear_velocity']['params']['std']
    assert left==right,'Unexpected live training config difference'
    baseline=yaml.load((paths['parent']/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    ref=read(paths['parent']/'evaluation.json')
    for arm in ('control','candidate'):
        assert yaml.load((paths[arm]/'env.yaml').read_text(),Loader=yaml.BaseLoader)==baseline
        current=read(paths[arm]/'evaluation.json')
        assert current['source_sha256']==ref['source_sha256'] and current['packages']==ref['packages']
    rows={k:metrics(p) for k,p in paths.items()}
    primary=pair['primary'];reg=pair['regression'];c=rows['candidate']
    gains={condition:all(
        c[condition]['linear_error']<=rows[reference][condition]['linear_error']-primary['error_reduction_min'] and
        c[condition]['signed_speed']>=rows[reference][condition]['signed_speed']+primary['signed_speed_gain_min']
        for reference in ('parent','control')) for condition in ('forward','backward')}
    checks=dict(stand_pass=c['stop']['aggregate_tracking_pass'],primary_improvement=any(gains.values()),
        posture_preserved=all(x['inward_fraction_after1s']<=reg['inward_fraction_max'] for x in c.values()),
        survival_preserved=all(x['survivors']/x['episodes']>=reg['survival_min'] for x in c.values()))
    regression=[]
    for reference in ('parent','control'):
        for condition in ('stop','forward','backward'):
            a,b=c[condition],rows[reference][condition]
            valid=(a['linear_error']<=b['linear_error']+reg['max_linear_error_increase'] and
                   a['yaw_error']<=b['yaw_error']+reg['max_yaw_error_increase'] and
                   a['signed_speed']>=b['signed_speed']-reg['max_signed_speed_drop'])
            regression.append(dict(reference=reference,condition=condition,passed=valid))
    checks['no_regression']=all(x['passed'] for x in regression)
    result=dict(status='passed',rows=rows,checks=checks,primary_per_command=gains,regression=regression,
        screen_passed=all(checks.values()),decision='complete-four-command-regression' if all(checks.values()) else 'stop-and-review-hypothesis',
        actual_training_configuration_single_factor=True,evaluation_configuration_equal=True,
        seven_command_acceptance=False,heldout_used=False,multi_seed_confirmed=False,
        protocol_sha256=sha(pair_path),script_sha256=sha(Path(__file__)),
        evaluations={k:dict(path=str(v),sha256=sha(v/'evaluation.json')) for k,v in paths.items()},
        scope='50-update single-seed diagnostic; no convergence or generalization claim')
    write(OUT/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
