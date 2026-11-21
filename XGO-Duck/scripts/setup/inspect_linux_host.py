"""Record a Linux/WSL host profile without installing or starting training.

Run with the system Python before bootstrapping a new native Ubuntu host.
Only XGO-Duck records are written. Existing project Python is inspected when
present; missing tools are recorded, not installed or silently substituted.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def capture(argv,timeout=25):
    try:
        result=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
        return dict(command=argv,exit_code=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
    except (FileNotFoundError,subprocess.TimeoutExpired) as error:
        return dict(command=argv,exit_code=None,error=str(error))


def main():
    if platform.system()!='Linux':
        raise SystemExit('Run this script inside WSL or native Linux.')
    project=Path(__file__).resolve().parents[2]
    dedicated=Path.home()/'XGO-Duck'
    os_release={}
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            key,value=line.split('=',1);os_release[key]=value.strip('"')
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemTotal','MemAvailable'): memory[key+'_kib']=int(value.split()[0])
    cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
    kind='wsl' if 'microsoft' in platform.release().lower() else 'native-linux'
    paths=[]
    for label,path in [('project',project),('dedicated_linux',dedicated)]:
        existing=path
        while not existing.exists(): existing=existing.parent
        disk=shutil.disk_usage(existing)
        paths.append(dict(label=label,path=str(path),exists=path.exists(),
                          disk_free_bytes=disk.free,disk_total_bytes=disk.total,
                          filesystem=capture(['findmnt','-T',str(existing),'-n','-o','FSTYPE,SOURCE'])))
    uv=dedicated/'tools/uv'
    uv_command=str(uv) if uv.is_file() else 'uv'
    checks=[capture(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader']),
            capture([uv_command,'--version']),capture(['git','--version'])]
    venv=dedicated/'workspace/xgoduck_rl/.venv/bin/python'
    python_check=None
    if venv.is_file():
        code='''import json,sys,importlib.metadata as m
import torch
names=['torch','mjlab','mujoco','mujoco-warp','warp-lang','onnxruntime','rsl-rl-lib']
print(json.dumps(dict(python=sys.version,packages={n:m.version(n) for n in names},cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))'''
        python_check=capture([str(venv),'-B','-c',code])
    record=dict(checked_at=datetime.now(timezone.utc).isoformat(),host_kind=kind,
        os_release=os_release,kernel=platform.release(),architecture=platform.machine(),
        cpu=cpu,logical_cpu_count=os.cpu_count(),memory=memory,paths=paths,
        system_python=sys.version,tool_checks=checks,project_python=python_check,
        training_started=False,packages_installed=False,drivers_modified=False,
        scope='Capability snapshot only; not a benchmark or native/WSL speed comparison')
    out=project/'records/deployment'/datetime.now().astimezone().strftime('%Y-%m-%d')
    out.mkdir(parents=True,exist_ok=True)
    file=out/f'host-profile-{kind}-{datetime.now().strftime("%H%M%S")}.json'
    with file.open('x') as f: json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(dict(profile=str(file),host_kind=kind,os=os_release.get('PRETTY_NAME'),
                          cpu=cpu,project_python_present=venv.is_file(),training_started=False),indent=2))


if __name__=='__main__':main()
