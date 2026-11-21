"""Index the exact policy files qualified by the nominal closed-loop evaluation."""

from pathlib import Path

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha


PROJECT = ROOT/'project.json'
KEY = 'nominal_roll_timestep_confirmation'
OUT = ROOT/'artifacts/delivery/nominal-continuous-simulation.json'
EXISTING = ROOT/'artifacts/delivery/routed-seven-command-simulation.json'
LOADER = ROOT/'records/reports/native-ppo-observation-preflight/delivery-checkpoint-cpu-loader-checked.json'
OFFICIAL = ROOT/'records/training-archive/delivery-weight-recheck.json'


def reference(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def main():
    assert not OUT.exists()
    project = read(PROJECT)
    task, stages = project['staged_training_request'], project['goal_execution']
    assert stages['status']=='active' and not stages['pause_requested_by_user'] and not task['pause_requested']
    assert not (task['simulation_running'] or task['learning_running'] or task['deployment_authorized_now'])
    heldout, repeat = (stages[key] for key in (KEY,'nominal_roll_timestep_repeat'))
    assert heldout['status']==repeat['status']=='reviewed' and heldout['passed'] and repeat['qualified']
    config, scoring, development = (ROOT/path for path in (heldout['protocol'],heldout['result'],repeat['result']))
    plan = read(config,heldout['protocol_sha256'])
    confirmation = read(scoring,heldout['result_sha256'])
    matched = read(development,repeat['result_sha256'])
    assert matched['advance_to_independent_confirmation'] and confirmation['nominal_confirmation_passed']
    assert confirmation['nominal_reserved_seed_used'] and not confirmation['low_friction_reserved_seed_used']
    assert confirmation['protocol_sha256']==sha(config)
    assert confirmation['seed']==plan['environment']['seed']==2256
    assert plan['environment']['num_envs']==confirmation['worlds']==896
    assert confirmation['controls']==plan['composition']['maximum_controls']==2125
    action = confirmation['result']
    assert action['video']['full_route_duration'] and action['video']['actual_seconds']==42.5
    assert sha(ROOT/action['video']['path'])==action['video']['sha256']
    assert action['getup']['functional_coverage_gate']
    assert set(action['commands'])==set(plan['seven_command']['groups'])
    for name, result in action['commands'].items():
        assert result['complete']>=plan['seven_command']['minimum_by_command'][name]
        assert result['original_command_score']['functional_development_pass']

    stored = read(EXISTING)['policy_roles']
    loader = {item['onnx']:item for item in read(LOADER)['results']}
    official = {item['path']:item for item in read(OFFICIAL)['files'] if item['role']=='official-reference'}
    all_official = {}
    for name, item in official.items():
        file, upstream = ROOT/name, ROOT/item['source']
        assert file.read_bytes()==upstream.read_bytes() and sha(file)==item['sha256']
        all_official[Path(name).name] = dict(archive=reference(file),source=reference(upstream))
    assert len(all_official)==4
    components = {}
    for role, policy in plan['models'].items():
        model = ROOT/policy['model']
        rt = reference(model)
        assert rt['sha256']==policy['model_sha256']
        if role=='official-forward-roll':
            assert official[rt['path']]['sha256']==rt['sha256']
            components[role] = dict(kind='official',onnx=rt)
            continue
        saved = stored[role]
        assert (saved['onnx']['path'],saved['onnx']['sha256'])==(rt['path'],rt['sha256'])
        component = dict(kind=saved['kind'],onnx=rt)
        if saved['kind']=='trained':
            state = saved['recoverable_checkpoint']
            audit = loader[rt['path']]
            assert audit['checkpoint']==state['path'] and audit['checkpoint_sha256']==state['sha256']
            assert audit['onnx_sha256']==rt['sha256'] and audit['actor_input_width']==61
            assert audit['actor_and_critic_exactly_loaded']
            assert audit['actor_optimizer']['loaded_state_exact'] and audit['critic_optimizer']['loaded_state_exact']
            assert sha(ROOT/state['path'])==sha(ROOT/state['source'])==state['sha256']
            component.update(recoverable_checkpoint=state,training_updates=audit['completed_updates'])
        else:
            assert official[rt['path']]['sha256']==rt['sha256']
        components[role] = component
    assert len(components)==7 and sum(x['kind']=='trained' for x in components.values())==len(loader)==5

    manifest = dict(status='verified-nominal-continuous-simulation',
        scope='First-episode 42.5 second fall, getup, stand, roll, stand and assigned command at friction 1.0.',
        return_to_hardware_runbook=reference(ROOT/'docs/runbooks/hardware_commissioning.md'),
        nominal_seed=2256, remaining_low_friction_seed=2246, worlds=confirmation['worlds'],
        timely_recovery_and_hold=action['getup']['overall']['timely_holds'],roll_and_landing=action['roll_complete'],
        complete=action['complete'],commands={name:result['complete'] for name,result in action['commands'].items()},
        motor_interval_s=plan['solver_comparison']['motor_update_s'],
        policy_interval_s=plan['solver_comparison']['control_update_s'],
        fine_solver_interval=plan['solver_comparison']['fine_solver_interval'],fine_solver_step_s=.002,
        nominal_protocol=reference(config),nominal_result=reference(scoring),development_pair=reference(development),
        checkpoint_loader_audit=reference(LOADER),official_sources=all_official,policy_roles=components,video=action['video'],
        hardware_io=False,hardware_deployment_qualified=False,
        robustness_verdict='pending-independent-disturbance-and-low-friction',
        single_onnx_contains_routing=False,export_script=reference(Path(__file__).resolve()))
    write(OUT,manifest)
    assert read(OUT)==manifest
    heldout['nominal_simulation_delivery'] = reference(OUT)
    write(PROJECT,project)
    print('NOMINAL_CONTINUOUS_SIMULATION_DELIVERY',sha(OUT),len(components),'policies',action['complete'],'complete',flush=True)


if __name__=='__main__':
    main()
