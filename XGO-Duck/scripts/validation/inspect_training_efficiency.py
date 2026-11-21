"""Read-only training timing summary plus bounded NVIDIA telemetry samples."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import statistics
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run_id')
p.add_argument('--samples',type=int,default=12)
args=p.parse_args()
root=Path(__file__).resolve().parents[2]
record=root/'records/experiments'/args.run_id
if record.parent!=root/'records/experiments' or not record.is_dir():
    raise SystemExit('Unknown run directory')
samples=[]
for i in range(args.samples):
    result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.current.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    values=[float(v.strip()) for v in result.stdout.strip().split(',')]
    samples.append(dict(zip(['gpu_percent','memory_controller_percent','vram_mib','power_w','temperature_c','sm_clock_mhz'],values),time=datetime.now(timezone.utc).isoformat()))
    if i+1<args.samples:time.sleep(1)
text=(record/'training.log').read_text(encoding='utf-8',errors='replace')
blocks=re.split(r'Learning iteration\s+(\d+)/\d+',text)
iterations=[]
for i in range(1,len(blocks),2):
    block=blocks[i+1]
    values={}
    for key,label in [('collection_seconds','Collection time'),('learning_seconds','Learning time'),('iteration_seconds','Iteration time'),('transitions_per_second','Steps per second')]:
        match=re.search(label+r':\s+([\d.]+)',block)
        if match:values[key]=float(match.group(1))
    if len(values)==4:iterations.append(dict(iteration=int(blocks[i]),**values))
# Exclude compilation/startup and 200 video frames around every 250 iterations.
ordinary=[v for v in iterations if v['iteration']>=20 and v['iteration']%250>=10][-100:]
summary={k:statistics.median(v[k] for v in ordinary) for k in ['collection_seconds','learning_seconds','iteration_seconds','transitions_per_second']} if ordinary else {}
gpu={k:{'min':min(v[k] for v in samples),'median':statistics.median(v[k] for v in samples),'max':max(v[k] for v in samples)} for k in samples[0] if k!='time'}
report={'run_id':args.run_id,'observed_at':datetime.now(timezone.utc).isoformat(),'latest_completed_iteration':iterations[-1]['iteration'] if iterations else None,'ordinary_iteration_count':len(ordinary),'ordinary_iteration_medians':summary,'gpu_telemetry':gpu,'samples':samples,'recent_iterations':iterations[-120:],'scope':'Read-only observational timings. No controlled configuration comparison, no speedup claim. GPU samples need not align with iteration boundaries.'}
out=record/('efficiency-'+datetime.now().strftime('%Y%m%dT%H%M%S')+'.json')
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['run_id','latest_completed_iteration','ordinary_iteration_count','ordinary_iteration_medians','gpu_telemetry']},indent=2))
print(out)
