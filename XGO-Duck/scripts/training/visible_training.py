"""Foreground official training with a loopback progress page and recorded video.

Run in the single visible project console. HTTP is an owned thread and closes
when training ends; the training child inherits the console and is never detached.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.parse import unquote

PROJECT = Path(__file__).resolve().parents[2]
WORK = Path.home() / 'XGO-Duck/workspace/xgoduck_rl'
PAGE = '''<!doctype html><meta charset="utf-8"><title>XGO-Duck 训练过程</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#101922;color:#edf3f8}h1{font-size:28px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}section{padding:18px;background:#1b2937;border-radius:12px}canvas{width:100%;height:220px}video{width:100%}pre{white-space:pre-wrap;font-size:12px;max-height:280px;overflow:auto}a{color:#7cd6ee}</style>
<h1>XGO-Duck · 训练过程</h1><p><a href="/report" target="_blank">打开全流程图文与视频报告</a></p><p id="state">正在读取实际训练记录…</p>
<p>终端执行训练；此页面每 10 秒读取日志。视频是训练时环境 0 的采样过程，可能包含重置，不能作为稳定行走验收。</p>
<div class="grid"><section><h2>平均奖励</h2><canvas id="reward" width="500" height="220"></canvas></section><section><h2>Episode 长度</h2><canvas id="length" width="500" height="220"></canvas></section></div>
<section><h2>仿真片段</h2><p id="caption">等待首段视频保存。</p><video id="video" controls muted loop></video><div id="videos"></div></section>
<section><h2>实时日志</h2><pre id="log"></pre></section>
<script>
function draw(id,points){let c=document.getElementById(id),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.fillStyle='#c4d4e4';if(!points.length){x.fillText('等待训练指标写入',20,40);return}let lo=Math.min(...points.map(p=>p[1])),hi=Math.max(...points.map(p=>p[1]));x.fillText('最新: '+points.at(-1)[1].toFixed(3)+' · iteration '+points.at(-1)[0],15,20);x.fillText('范围: '+lo.toFixed(2)+' — '+hi.toFixed(2),15,210);x.strokeStyle='#63dbb0';x.beginPath();points.forEach((p,i)=>{let px=15+470*(p[0]-points[0][0])/Math.max(1,points.at(-1)[0]-points[0][0]),py=185-145*(p[1]-lo)/Math.max(1e-9,hi-lo);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}
async function refresh(){try{let s=await(await fetch('/state')).json();document.getElementById('state').textContent=s.run_id+' · '+s.status+' · '+s.iterations+' 次计划迭代 · seed '+s.seed;draw('reward',s.reward);draw('length',s.length);document.getElementById('log').textContent=s.log;let list=document.getElementById('videos');list.replaceChildren();s.videos.forEach(n=>{let a=document.createElement('a');a.href='/video/'+encodeURIComponent(n);a.textContent=n+' ';a.onclick=e=>{e.preventDefault();document.getElementById('video').src=a.href;document.getElementById('caption').textContent=n};list.append(a)});let v=document.getElementById('video');if(!v.getAttribute('src')&&s.videos.length){let n=s.videos.at(-1);v.src='/video/'+encodeURIComponent(n);document.getElementById('caption').textContent=n}}catch(e){document.getElementById('state').textContent+=' · 连接已结束或暂不可用；结果保存在实验目录'}}refresh();setInterval(refresh,10000);
</script>'''

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['baseline','candidate'],required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preflight',action='store_true')
    continuation=p.add_mutually_exclusive_group()
    continuation.add_argument('--resume',action='store_true',help='Continue the declared paused baseline checkpoint after explicit user resume')
    continuation.add_argument('--extend',action='store_true',help='Run the separately frozen 5000-to-7500 official baseline stage')
    continuation.add_argument('--live-resume',action='store_true',help='Resume the authorized saved extension with an inspect-only native viewer')
    args=p.parse_args()
    # Honor the user's pause even when this Python file is invoked directly,
    # before creating a run folder, dashboard or training subprocess.
    project=json.loads((PROJECT/'project.json').read_text(encoding='utf-8-sig'))
    if project['formal_training']['status'].startswith('paused'):
        raise SystemExit('Training is paused by user. No training started.')
    if project.get('gait_anomaly',{}).get('long_training_resume_requires_diagnostic_resolution'):
        raise SystemExit('Resolve and verify the registered gait anomaly before starting further learning.')
    if args.mode=='candidate' and not project['formal_training']['stable_gait_verified']:
        raise SystemExit('Stable baseline gait must be verified before candidate comparison.')
    # A resumed baseline and a fresh candidate are not a reward-only comparison.
    # Require the paired host/start/restart review before creating any candidate run.
    if args.mode=='candidate' and project['stage_plan'].get('comparison_start_conditions_verified') is not True:
        raise SystemExit('Matched comparison start conditions must be verified before candidate training.')
    if (args.resume or args.extend or args.live_resume) and (args.mode!='baseline' or args.preflight):
        raise SystemExit('Continuation is only supported for the formal baseline.')
    protocol=json.loads((PROJECT/'configs/experiments/formal_velocity_tracking.json').read_text())
    smoke=json.loads((PROJECT/'configs/experiments/velocity_tracking.json').read_text())
    source=WORK/smoke['change']['file']
    expected=smoke['change']['source_sha256' if args.mode=='baseline' else 'candidate_sha256']
    if digest(source)!=expected:
        raise SystemExit('Source differs from requested group; restore/apply the recorded patch first.')
    if digest(WORK/'uv.lock')!='0ef769135d86e63c03663b2df570c0217217b541ab234460a879dff8fb260fa9':
        raise SystemExit('Official dependency lock mismatch')
    if args.seed not in protocol['training']['seeds']:
        raise SystemExit('Seed is outside the fixed protocol')
    phase='extension' if (args.extend or args.live_resume) else ('continuation' if args.resume else ('video_preflight' if args.preflight else 'formal'))
    run_id=f'{args.mode}_{phase}_{datetime.now().strftime("%Y%m%dT%H%M%S")}_s{args.seed}'
    resume_plan=None
    extension_evidence=None
    if args.resume:
        from resume_state import load_plan
        resume_plan=load_plan(PROJECT,WORK)
        if args.seed!=resume_plan['seed']:
            raise SystemExit('Continuation must retain the saved baseline seed.')
    if args.extend or args.live_resume:
        if args.live_resume:
            from live_resume import prepare
        else:
            from extend_official import prepare
        _,extension_protocol,resume_plan,extension_evidence=prepare(run_id)
        if args.seed!=resume_plan['seed']:
            raise SystemExit('Extension must retain the saved baseline seed.')
        # One execution per frozen extension. A failed/interrupted run is kept
        # for explicit recovery planning, never silently restarted from 4999.
        protocol_key='live_protocol' if args.live_resume else 'extension_protocol'
        for prior in (PROJECT/'records/experiments').glob('baseline_extension_*/training-request.json'):
            previous=json.loads(prior.read_text())
            if previous.get('extension_source_sha256',{}).get(protocol_key)==extension_evidence['source_sha256'][protocol_key]:
                raise SystemExit(f'This extension already has a run: {prior.parent}. Inspect it before any new training.')
    record=PROJECT/'records/experiments'/run_id
    record.mkdir(parents=True,exist_ok=False)
    (record/'videos').mkdir()
    iterations=20 if args.preflight else protocol['training']['max_iterations']
    if resume_plan:
        iterations=resume_plan['remaining_updates']
    num_envs=64 if args.preflight else protocol['training']['num_envs']
    if resume_plan:
        num_envs=resume_plan['num_envs']
    command=['uv','run','--locked','--offline','train',protocol['task'],
             '--env.scene.num-envs',str(num_envs),'--agent.max-iterations',str(iterations),
             '--agent.seed',str(args.seed),'--agent.run-name',run_id,
             '--agent.logger','tensorboard','--agent.save-interval','250',
             '--video','True','--video-length','200','--video-interval','6000',
             '--env.viewer.width','960','--env.viewer.height','720',
             '--env.viewer.distance','1.2','--env.viewer.elevation','-20',
             '--env.viewer.max-extra-envs','0']
    if resume_plan:
        command=['uv','run','--locked','--offline','python','-B',
                 str(PROJECT/'scripts/training'/('live_resume.py' if args.live_resume else ('extend_official.py' if args.extend else 'resume_official.py'))),
                 '--run-id',run_id,'--record-dir',str(record)]
    request={'run_id':run_id,'mode':args.mode,'phase':phase,'seed':args.seed,
             'num_envs':num_envs,'iterations':iterations,'command':command,
             'started_at':datetime.now(timezone.utc).isoformat(),'task_source_sha256':digest(source),
             'lock_sha256':digest(WORK/'uv.lock'),'status':'starting'}
    if resume_plan:
        request['resume_plan']=resume_plan
        request['total_iterations_including_saved']=resume_plan['total_updates']
        request['resume_source_sha256']={}
        for name in ('resume_state.py','resume_official.py'):
            file=Path(__file__).with_name(name)
            request['resume_source_sha256'][name]=digest(file)
            shutil.copy2(file,record/name)
    if extension_evidence:
        request['extension_source_sha256']=extension_evidence['source_sha256']
        (record/'sources').mkdir()
        for key,path in extension_evidence['source_paths'].items():
            target=record/'sources'/f'{key}.txt'
            shutil.copy2(path,target)
            assert digest(target)==extension_evidence['source_sha256'][key]
        shutil.copy2(PROJECT/'configs/experiments'/('live_training_resume.json' if args.live_resume else 'baseline_extension.json'),record/'extension-protocol.json')
        if args.live_resume:
            request['visualization']={'kind':'native-live-training-state','environment_index':0,
                'max_wall_fps':extension_protocol['max_wall_fps'],'extra_environment_steps':0,
                'mouse_perturbations_applied':False,'restart_boundary':True}
    (record/'training-request.json').write_text(json.dumps(request,indent=2))
    (record/'protocol.json').write_text(json.dumps(protocol,indent=2))
    # Preserve the executable experiment explanation with the run, so later
    # edits to the entry script cannot change what this experiment claimed.
    entry = os.environ.get('XGODUCK_EXPERIMENT_ENTRY')
    if entry:
        entry_path = Path(entry).resolve()
        if not entry_path.is_relative_to(PROJECT/'scripts/training/experiments'):
            raise SystemExit('Experiment entry is outside the project experiment directory')
        shutil.copy2(entry_path,record/'experiment-entry.sh')
    shutil.copy2(Path(__file__),record/'training-runner.py')
    log_path=record/'training.log'
    log_root=WORK/'logs/rsl_rl/xgoduck_velocity'
    state={'run_id':run_id,'status':'starting','iterations':iterations,'seed':args.seed}
    log_dir=None
    accumulator=None
    def snapshot():
        nonlocal log_dir,accumulator
        if log_dir is None:
            matches=list(log_root.glob('*_'+run_id))
            if len(matches)==1:
                log_dir=matches[0]
        result={**state,'reward':[],'length':[],'videos':[],'log':''}
        if log_path.exists():
            with log_path.open('rb') as f:
                f.seek(max(0,log_path.stat().st_size-12000))
                result['log']=f.read().decode('utf-8',errors='replace')
        if log_dir:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            if accumulator is None:
                accumulator=EventAccumulator(str(log_dir),size_guidance={'scalars':6000})
            accumulator.Reload()
            tags=accumulator.Tags().get('scalars',[])
            for key,tag in [('reward','Train/mean_reward'),('length','Train/mean_episode_length')]:
                if tag in tags:
                    result[key]=[[e.step,e.value] for e in accumulator.Scalars(tag)]
            result['scalar_tags']=tags
            # Numerical step order keeps step-12000 after step-6000. Filename
            # order would make the page select an older clip as the latest one.
            video_key=lambda f: (int(f.stem.rsplit('-',1)[-1]) if f.stem.rsplit('-',1)[-1].isdigit() else -1,f.name)
            result['videos']=[f.name for f in sorted((log_dir/'videos/train').glob('*.mp4'),key=video_key) if f.stat().st_size>0]
            result['log_dir']=str(log_dir)
        return result
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*unused): pass
        def do_GET(self):
            try:
                if self.path=='/': body=PAGE.encode();kind='text/html; charset=utf-8'
                elif self.path=='/report': body=(PROJECT/'docs/reports/development_report.html').read_bytes();kind='text/html; charset=utf-8'
                elif self.path=='/state': body=json.dumps(snapshot()).encode();kind='application/json'
                elif self.path.startswith('/video/') and log_dir:
                    name=unquote(self.path[len('/video/'):])
                    if Path(name).name!=name or not name.endswith('.mp4'):
                        self.send_error(404);return
                    file=log_dir/'videos/train'/name
                    if not file.is_file():self.send_error(404);return
                    body=file.read_bytes();kind='video/mp4'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind)
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as exc:
                self.send_error(500,str(exc))
    server=ThreadingHTTPServer(('127.0.0.1',8765),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'VISIBLE TRAINING: {run_id}\nDashboard: http://localhost:8765\nCOMMAND: {command}',flush=True)
    (PROJECT/'workspace/console/current-training.json').write_text(json.dumps({**request,'record_dir':str(record),'url':'http://localhost:8765'},indent=2))
    process=None
    try:
        with log_path.open('w',buffering=1) as log:
            process=subprocess.Popen(command,cwd=WORK,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            state['status']='running'
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line)
            code=process.wait()
        state['status']='training-finished' if code==0 else 'failed'
    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        code=130;state['status']='interrupted'
    except Exception as error:
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        code=1;state['status']='failed'
        request['error']=repr(error)
        print(f'Training supervisor error: {error}',flush=True)
    finally:
        server.shutdown();server.server_close()
    request.update(status=state['status'],exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
    final=snapshot()
    if log_dir:
        request['log_dir']=str(log_dir)
        shutil.copytree(log_dir/'params',record/'params',dirs_exist_ok=True)
        for file in (log_dir/'videos/train').glob('*.mp4'):
            shutil.copy2(file,record/'videos'/file.name)
        request['checkpoints']=[{'path':str(f),'sha256':digest(f)} for f in sorted(log_dir.glob('model_*.pt'))]
    (record/'training-result.json').write_text(json.dumps(request,indent=2))
    (record/'progress.json').write_text(json.dumps(final,indent=2))
    (record/'exit-code.txt').write_text(str(code)+'\n')
    (record/'dashboard.html').write_text(PAGE.replace("fetch('/state')","fetch('progress.json')").replace("'/video/'","'videos/'"))
    print(f'FINISHED: {record}; exit={code}',flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
