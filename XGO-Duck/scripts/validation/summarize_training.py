"""Verify saved training parameters and checkpoint, and preserve the evidence."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-id', required=True)
parser.add_argument('--checkpoint', type=Path, required=True)
args = parser.parse_args()
project = Path(__file__).resolve().parents[2]
record = project / 'records/experiments' / args.run_id
request = json.loads((record / 'training-request.json').read_text())
definition = json.loads((record / 'experiment-definition.json').read_text())
checkpoint = args.checkpoint.resolve(strict=True)
assert request['run_name'] == args.run_id and args.run_id in checkpoint.parent.name
assert checkpoint.is_relative_to(Path(request['working_copy']) / 'logs')
assert (record / 'exit-code.txt').read_text().strip() == '0'
agent = yaml.load((checkpoint.parent / 'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
env = yaml.load((checkpoint.parent / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader)
smoke = request['smoke']
assert int(agent['seed']) == smoke['seed']
assert int(agent['max_iterations']) == smoke['max_iterations']
assert int(env['scene']['num_envs']) == smoke['num_envs']
assert agent['run_name'] == args.run_id
weight = float(env['rewards']['track_linear_velocity']['weight'])
assert weight == definition['change'][request['run_kind']]
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
assert state['iter'] == smoke['max_iterations'] - 1
def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)
checked = list(tensors(state))
assert checked and all(torch.isfinite(tensor).all().item() for tensor in checked)
params = record / 'params'
params.mkdir(exist_ok=True)
for filename in ['agent.yaml', 'env.yaml']:
    shutil.copy2(checkpoint.parent / 'params' / filename, params / filename)
report = {
    'checked_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
    'run_id': args.run_id, 'run_kind': request['run_kind'], 'smoke': smoke,
    'checkpoint': str(checkpoint), 'checkpoint_iteration': state['iter'],
    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'checkpoint_tensors_checked': len(checked), 'all_checkpoint_tensors_finite': True,
    'saved_tracking_reward_weight': weight,
    'params_sha256': {name: hashlib.sha256((params/name).read_bytes()).hexdigest() for name in ['agent.yaml', 'env.yaml']},
    'scope': 'Training pipeline and saved configuration; no gait quality or hardware acceptance',
}
(record / 'training-summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
