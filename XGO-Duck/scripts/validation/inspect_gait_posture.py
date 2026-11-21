"""CPU-only posture review of the two closed grouped simulation studies.

Report actual trunk height, tilt and signed left-minus-right foot separation
along the trunk's local Y axis. This is a site-center proxy, not collision
mesh clearance. Do not overwrite the earlier functional decisions.
"""
from pathlib import Path
import json
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def posture(trace):
    w,x,y,z=np.moveaxis(trace['root_quaternion'],-1,0)
    body_y=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
    separation=np.sum((trace['foot_positions'][:,:,0]-trace['foot_positions'][:,:,1])*body_y,axis=-1)
    tilt=np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)))
    return separation,tilt


def main():
    rows=[]
    for stage in ('straight-parallel','turn-feedback'):
        progress=read(ROOT/f'records/reports/{stage}/progress.json');assert progress['status']=='collected-and-verified'
        folder=Path(progress['path']);report=read(folder/'evaluation.json')
        for item in report['conditions']:
            path=folder/f'{item["condition"]}-trace.npz';assert sha(path)==item['trace_sha256']
            with np.load(path) as trace:
                sep,tilt=posture(trace);mask=trace['active'].copy();mask[:50]=False
                rows.append(dict(stage=stage,condition=item['condition'],trace_sha256=sha(path),
                    height_quantiles=np.quantile(trace['height'][mask],[.01,.5,.99]).tolist(),
                    tilt_deg_quantiles=np.quantile(tilt[mask],[.01,.5,.99]).tolist(),
                    foot_center_separation_quantiles=np.quantile(sep[mask],[.01,.5,.99]).tolist(),
                    fraction_separation_below_15mm=float((sep[mask]<.015).mean()),
                    fraction_tilt_above_25deg=float((tilt[mask]>25.).mean())))
    out=ROOT/'records/reports/seven-command';out.mkdir(exist_ok=True)
    result=dict(status='diagnostic-only',rows=rows,script_sha256=sha(Path(__file__)),
        quantiles=[.01,.5,.99],warmup_seconds=1.,environment_steps=0,earlier_verdicts_unchanged=True,
        geometry_scope='Signed site-center separation in trunk frame; not actual collision mesh clearance')
    write(out/'prior-posture-review.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
