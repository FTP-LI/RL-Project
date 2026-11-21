"""Recover the roll-run index from closed, already saved acquisition artifacts."""

import json

import numpy as np
import torch

import run_roll_active_lateral as study
from verify_continuous_forward_yaw_pair import joined
from verify_continuous_roll_projected_entry import count
from verify_continuous_roll_stand_walk import arrays


def main():
    p = study.checked()
    read, sha, write = study.read, study.sha, study.write
    assert not torch.cuda.is_initialized()
    folder = study.ROOT/p['record']
    protocol = folder/'protocol.json'
    assert sha(protocol) == sha(study.PLAN) == read(folder/'command.json')['protocol_sha256']
    state_path = folder/'state.json'
    runtime_path = folder/'runtime-contract.json'
    observer_path = folder/'windows-view-verification.json'
    exit_path = study.OUT/'sample-exit.json'
    log = study.OUT/'sample.log'
    runtime, state, view, exit_status = map(read, (runtime_path, state_path, observer_path, exit_path))
    assert exit_status['exit_code'] == 1 and state['status'] == 'stopped-review-required'
    assert state['error'] == 'NameError("name \'PLAN\' is not defined")'
    assert 'NameError: name \'PLAN\' is not defined' in log.read_text(encoding='utf-8')
    assert runtime['status'] == 'closed' and runtime['interface_checks'] == 1
    assert not runtime['capacity_reached'] and not runtime['extra_observation_or_render_steps']
    assert view['status'] == 'review-required' and view['observations'][-1]['status'] == state['status']
    assert view['observations'][-1]['controls'] == state['controls'] == runtime['control_steps']
    assert view['running_snapshots_with_original_frame'] > 0 and view['snapshots_with_complete_clip'] > 0
    assert not (folder/'shared-stop.json').exists()
    plan = p['perturbation']
    shared, horizon = plan['shared'], p['maximum_controls']
    prefix_path, commands_path = (folder/name for name in ('shared-trace.npz','shared-commands.npz'))
    prefix, commands = arrays(prefix_path), arrays(commands_path)
    assert len(prefix['active']) == shared+1 and len(commands['phase_used']) == shared
    paths = [study.PLAN, protocol, log, exit_path, state_path, runtime_path, observer_path,
        folder/'fork.json', prefix_path, commands_path]
    branches = {}
    for role in plan['roles']:
        branch = folder/'clips'/role
        trace_path, command_path, clip_path = (branch/name for name in ('trace.npz','commands.npz','clips.json'))
        suffix, after = arrays(trace_path), arrays(command_path)
        controls = len(after['phase_used'])
        assert controls > 0 and len(suffix['active']) == controls+1
        for name in prefix:
            np.testing.assert_array_equal(prefix[name][-1], suffix[name][0])
        last = shared+controls
        _, stop = count(p, joined(prefix, suffix, 1), joined(commands, after, 0))
        assert last <= horizon and (last == horizon or stop is not None and stop['control'] == last)
        clip, = read(clip_path)
        assert clip['common_steps'][-1] == last and clip['complete_clip'] == (last == horizon)
        video = branch/clip['file']
        metadata = dict(clip, path=video.relative_to(folder).as_posix(), sha256=sha(video))
        branches[role] = dict(endpoint=last, controls=controls, stop=stop, video=metadata)
        paths.extend((trace_path, command_path, clip_path, video, branch/'endpoint-contact.json',
            branch/'endpoint-contact.npz', branch/'restored-fork.json', branch/'injection.npz', branch/'injection.json'))
    controls = shared + sum(row['controls'] for row in branches.values())
    assert controls == runtime['control_steps'] <= plan['maximum_unique_controls']
    assert controls*p['worlds'] == runtime['control_transitions'] == state['transitions']
    assert controls*5 == runtime['physics_substeps']
    assert sha(folder/view['observations'][-1]['latest_clip']['path']) == view['observations'][-1]['latest_clip']['sha256']
    assert not torch.cuda.is_initialized()
    target = folder/'evaluation.json'
    receipt = study.OUT/'index-recovery.json'
    assert not receipt.exists()
    recovered = dict(status='collected-review-required', protocol_sha256=sha(study.PLAN), shared=shared,
        controls=controls, transitions=runtime['control_transitions'], substeps=runtime['physics_substeps'],
        physical_resets=1, branches=branches, learning_updates=0, heldout_used=False, index_reconstructed_offline=True)
    if target.exists():
        assert read(target) == recovered
    else:
        with target.open('x', encoding='utf-8') as stream:
            json.dump(recovered, stream, indent=2)
    write(receipt, dict(status='reconstructed-after-final-index-nameerror', physical_substeps=0, original_exit_code=1,
        index_sha256=sha(target), source_sha256={study.relative(path):sha(path) for path in paths},
        final_endpoints={role:row['endpoint'] for role,row in branches.items()},
        original_observer_status=view['status'], running_views=view['running_snapshots_with_original_frame']))
    print('ROLL_INDEX_RECOVERED', controls, runtime['control_transitions'],
        {role:(row['endpoint'],row['stop']) for role,row in branches.items()}, flush=True)


if __name__ == '__main__':
    main()
