"""Check that block statistics keep net bias and cannot wash out failures."""
from pathlib import Path
import hashlib
import json
import numpy as np
from planar_block_metrics import block_statistic

ROOT=Path(__file__).resolve().parents[2]


def main():
    t=np.arange(1000)*.02;active=np.ones((1000,2),bool)
    v=np.zeros((1000,2,2));v[:,:,0]=.15
    a=block_statistic(v,active,[.15,0,0]);np.testing.assert_allclose(a['mean_error_per_episode'],0,atol=1e-14)
    v[:,:,1]=(.18*np.sin(2*np.pi*2*t))[:,None]
    b=block_statistic(v,active,[.15,0,0]);np.testing.assert_allclose(b['mean_error_per_episode'],0,atol=1e-14)
    instant=float(np.linalg.norm(v-np.array([.15,0]),axis=2).mean());assert instant>.1
    v[:,:,1]=.1;c=block_statistic(v,active,[.15,0,0]);np.testing.assert_allclose(c['mean_error_per_episode'],.1,atol=1e-14)
    v[:]=0;d=block_statistic(v,active,[.15,0,0]);np.testing.assert_allclose(d['mean_error_per_episode'],.15,atol=1e-14)
    active[500:]=False;e=block_statistic(v,active,[.15,0,0]);assert (e['mean_error_per_episode']>.3).all()
    active[700:]=True
    try:block_statistic(v,active,[.15,0,0])
    except AssertionError:pass
    else:raise AssertionError('A reset revived a failed first episode')
    result=dict(status='passed',cases=['constant tracking','periodic zero-mean sway','persistent drift',
        'stationary-forward remains wrong','missing blocks penalized','reset cannot revive first episode'],
        cyclic_example_instantaneous_error=instant,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric_sha256=hashlib.sha256((ROOT/'scripts/validation/planar_block_metrics.py').read_bytes()).hexdigest(),
        scope='CPU metric mechanics, not task acceptance or policy improvement')
    out=ROOT/'records/reports/planar-filter';out.mkdir(exist_ok=True)
    (out/'block-metric-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
