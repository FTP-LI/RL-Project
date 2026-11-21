"""Read the actual released training and scoring contracts without building a scene."""

import numpy as np
import yaml

import evaluate_forward_official_stance as stance
import run_roll_stand_forward_diagnostic as direct


ROOT = direct.ROOT
REPORT = ROOT / 'records/reports/forward-action-sampling'
OUT = REPORT / 'environment-contract.json'


def checked_archive():
    project = direct.read(direct.P)
    release = project['goal_execution'][stance.KEY]
    plan_path = stance.PLANS / (stance.CASE + '.json')
    plan = direct.read(plan_path)
    score_ref = release['scored'][stance.CASE]
    receipt_path = ROOT / score_ref['path']
    receipt = direct.read(receipt_path)
    run = ROOT / 'records/evaluations' / receipt['run_id']
    manifest = direct.read(run/'guarded-evaluation.json')
    direct.require(release['status'] == 'reviewed' and direct.sha(receipt_path) == score_ref['sha256']
                   and direct.sha(plan_path) == receipt['protocol_sha256'] and receipt['full_route']
                   and direct.sha(run/'trace.npz') == manifest['trace_sha256']
                   and direct.sha(run/'commands.npz') == manifest['command_sha256'], 'Released evaluation evidence changed')

    trained_path = stance.pair.training.REPORT / 'training-check.json'
    trained = direct.read(trained_path)['runs']['candidate']
    protocol = ROOT / trained['protocol']
    training = direct.read(protocol)
    parent = ROOT / training['parent']
    zero = parent.parent / 'candidate-initial-state.npz'
    completed = ROOT / trained['record']
    direct.require(direct.sha(ROOT/trained['checkpoint']) == trained['checkpoint_sha256']
                   and direct.sha(protocol) == trained['protocol_sha256']
                   and direct.sha(parent) == training['parent_sha256']
                   and trained['model_sha256'] == plan['walking']['model_sha256'], 'Released training evidence changed')
    return plan, run, completed, zero, (plan_path, receipt_path, protocol, trained_path)


def yaml_data(node):
    # Compose the archive as YAML data. Tagged Python callables stay names.
    if isinstance(node, yaml.MappingNode):
        return {key.value: yaml_data(value) for key, value in node.value}
    if isinstance(node, yaml.SequenceNode):
        return [yaml_data(value) for value in node.value]
    kind = node.tag.rsplit(':', 1)[-1]
    if kind == 'null':
        return None
    if kind == 'bool':
        return node.value.lower() == 'true'
    if kind in ('int', 'float'):
        return float(node.value)
    prefix = 'tag:yaml.org,2002:python/name:'
    return node.tag.removeprefix(prefix) if node.tag.startswith(prefix) else node.value


def policy_config(document):
    actor = document['observations']['actor']
    terms = actor['terms']
    axes = {'base_ang_vel': (0, 3), 'projected_gravity': (3, 6), 'joint_pos': (6, 20), 'joint_vel': (20, 34)}
    direct.require(tuple(terms) == (*axes, 'actions', 'command', 'head_command', 'body_command'),
                   'Archived input layout differs')
    corruption = actor['enable_corruption']
    direct.require(isinstance(corruption, bool), 'Archived input corruption flag is not boolean')
    noise = {}
    for name, span in axes.items():
        term = terms[name]
        spec = term['noise']
        direct.require(spec['operation'] == 'add' and term['scale'] is None and term['clip'] is None
                       and spec['n_min'] == -spec['n_max'] and spec['n_max'] > 0,
                       'Unexpected additive-noise contract in ' + name)
        noise[name] = dict(slice=list(span), declared_range=[spec['n_min'], spec['n_max']], applied=corruption)
    return dict(input_corruption=corruption, observed_terms=list(terms), noise=noise)


def start_stats(qpos, actor):
    direct.require(qpos.shape == (len(actor), 21) and actor.shape[1] == 61
                   and np.isfinite(qpos).all() and np.isfinite(actor).all(), 'Archived startup arrays differ')
    root_tilt = np.rad2deg(np.arccos(np.clip(1-2*(qpos[:, 4]**2+qpos[:, 5]**2), -1., 1.)))
    history = np.sqrt(np.square(actor[:, 34:48]).mean(axis=1))
    return dict(worlds=len(actor), root_height_median_m=float(np.median(qpos[:, 2])),
                root_tilt_median_deg=float(np.median(root_tilt)),
                action_history_rms_median_rad=float(np.median(history)),
                maximum_abs_head_body_command=float(np.abs(actor[:, 51:]).max()))


def main():
    plan, evaluation, training, zero, bound = checked_archive()
    train_cfg, eval_cfg = training/'environment.yaml', evaluation/'env.yaml'
    train, review = [yaml_data(yaml.compose(path.read_text(encoding='utf-8'))) for path in (train_cfg, eval_cfg)]
    trained, scored = policy_config(train), policy_config(review)
    direct.require(trained['input_corruption'] and not scored['input_corruption'], 'Expected a training input-corruption mismatch')
    start = plan['initial_stand_controls']
    command_path, trace_path = evaluation/'commands.npz', evaluation/'trace.npz'
    with np.load(zero, allow_pickle=False) as data, np.load(command_path, allow_pickle=False) as commands, \
            np.load(trace_path, allow_pickle=False) as trace:
        train_start = start_stats(data['qpos'], data['actor'])
        eval_start = start_stats(trace['qpos'][start], commands['actor_input'][start])
    compare = dict(
        training=dict(trained, seed=int(train['seed']), auto_reset=train['auto_reset'],
                      episode_length_s=train['episode_length_s'], start=train_start),
        evaluation=dict(scored, seed=int(review['seed']), auto_reset=review['auto_reset'],
                        episode_length_s=review['episode_length_s'], start=eval_start, forward_entry_control=start),
        equal_declared_simulation_parameters=train['sim'] == review['sim'],
        equal_declared_control_decimation=train['decimation'] == review['decimation'],
        equal_declared_robot_configuration=train['scene']['entities']['robot'] == review['scene']['entities']['robot'],
        training_only_events=sorted(train['events'].keys() - review['events'].keys()),
        evaluation_only_events=sorted(review['events'].keys() - train['events'].keys()),
        shared_events_with_different_values=sorted(name for name in train['events'].keys() & review['events'].keys()
                                                   if train['events'][name] != review['events'][name]),
        sampling_scope='Archived profiles and realized startup distributions from their original seeds',
        sources={direct.rel(path): direct.sha(path) for path in (train_cfg, eval_cfg, zero, command_path, trace_path, *bound)})
    direct.put(OUT, compare)
    print('FORWARD_ENVIRONMENT_CONTRACT', train_start, eval_start, compare['evaluation_only_events'], flush=True)


if __name__ == '__main__':
    main()
