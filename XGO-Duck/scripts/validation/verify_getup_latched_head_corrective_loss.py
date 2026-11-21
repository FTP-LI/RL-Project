"""Run the native correction loss on CPU and CUDA without an optimizer or simulator."""

from pathlib import Path

import numpy as np
import torch

from run_getup_teacher_head_evaluation import ROOT, base, pinned, read, sha, write
from getup_leg_head_training import restrict_actor_to_leg_output
from getup_recovery_retention_ppo import retention_indices
from getup_teacher_anchor import mean_anchor_loss
from review_getup_archived_policy_teacher_timing import actor as read_onnx_actor


PLAN = ROOT / 'configs/experiments/getup_latched_head_corrective_loss_preflight.json'


def check_device(device, dtype, policy, joints, checkpoint, replay, draws, p):
    actor = base.shared.actor_model(policy, device=device).to(dtype=dtype)
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    before = base.shared.signature(actor.state_dict())
    layer, legs = restrict_actor_to_leg_output(actor, joints)
    x, control, candidate = (torch.as_tensor(replay[key], device=device, dtype=dtype) for key in
                             ('observations', 'control_targets', 'candidate_targets'))
    with torch.inference_mode():
        error = float((actor(base.shared.td('actor', x)) - control).abs().max())
    assert error <= p['maximum_control_target_error_rad'], (device, error)
    expected = {id(layer.weight), id(layer.bias)}
    assert {id(value) for value in actor.parameters() if value.requires_grad} == expected
    outputs, gradients = [], {}
    for index, ids in enumerate(draws):
        ids = ids.to(device)
        for arm, target in (('control', control), ('candidate', candidate)):
            actor.zero_grad(set_to_none=True)
            loss = mean_anchor_loss(actor(base.shared.td('actor', x[ids])), target[ids], p['initial_std_rad'])
            weighted = p['auxiliary_weight'] * loss
            weighted.backward()
            assert all(value.grad is None for value in actor.parameters() if id(value) not in expected)
            assert torch.count_nonzero(layer.weight.grad[~legs]) == torch.count_nonzero(layer.bias.grad[~legs]) == 0
            gradient = torch.cat((layer.weight.grad.flatten(), layer.bias.grad)).detach().cpu()
            norm = float(torch.linalg.vector_norm(gradient))
            assert torch.isfinite(gradient).all()
            if arm == 'control':
                assert norm < p['minimum_candidate_loss_gradient_norm']
            else:
                assert norm >= p['minimum_candidate_loss_gradient_norm']
            outputs.append(dict(arm=arm, minibatch=index, loss=float(loss.detach()),
                                weighted_loss=float(weighted.detach()), gradient_norm=norm))
            gradients[index, arm] = gradient
    assert base.shared.signature(actor.state_dict()) == before
    return dict(control_target_error_rad=error, gradients=outputs, only_leg_rows=True,
                state_unchanged=True), gradients


def main():
    p = read(PLAN)
    project = read(ROOT / 'project.json')
    assert project['current_phase'] == p['stage'] and ROOT / project['goal_execution']['current_protocol'] == PLAN
    assert not any(p[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    review_path = pinned(p['replay'])
    pinned(p['first_attempt']['log'])
    receipt = review_path.parent / 'native-loss-preflight.json'
    assert not receipt.exists()
    review = read(review_path)
    assert review['admitted_to_training_preparation'] and not review['heldout_used']
    pool = review['data']
    pool_path = pinned((pool['path'], pool['sha256']))
    cells = read(pinned((review['cells'], review['cells_sha256'])))
    assert len(cells) == 24 and sorted(row for cell in cells.values() for row in cell) == list(range(900))
    plan = read(pinned(p['source_protocol']))
    _, joints = read_onnx_actor(plan, 'candidate')
    checkpoint_path = pinned(p['checkpoint'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert not checkpoint['actor_optimizer_state_dict']['state']
    draws = [retention_indices(cells, p['sampling_seed'], minibatch, p['samples_per_cell']) for minibatch in range(2)]
    assert not torch.equal(*draws)
    with np.load(pool_path, allow_pickle=False) as replay:
        for ids in draws:
            assert len(ids) == 96 and np.count_nonzero(replay['kind'][ids.numpy()] == 'corrective') == 48
        args = plan['policies']['candidate'], joints, checkpoint, replay, draws, p
        oracle, reference = check_device('cpu', torch.float64, *args)
        cpu, first = check_device('cpu', torch.float32, *args)
        assert not torch.cuda.is_initialized() and torch.cuda.is_available()
        cuda, second = check_device('cuda:0', torch.float32, *args)
    for report, actual in ((cpu, first), (cuda, second)):
        for row in report['gradients']:
            key = row['minibatch'], row['arm']
            error = actual[key].double() - reference[key]
            row['maximum_element_error_against_float64'] = float(error.abs().max())
            assert row['maximum_element_error_against_float64'] <= p['maximum_float32_element_error_against_float64']
            if row['arm'] == 'candidate':
                relative = float(error.norm() / reference[key].norm())
                assert relative <= p['maximum_candidate_relative_gradient_l2_error_against_float64']
                row['relative_gradient_l2_error_against_float64'] = relative
    assert sha(checkpoint_path) == p['checkpoint'][1]
    write(receipt, dict(status='passed-native-cpu-cuda-gradient-only', protocol_sha256=sha(PLAN),
                        script_sha256=sha(Path(__file__)), checkpoint_sha256=sha(checkpoint_path),
                        replay_sha256=sha(pool_path), cpu=cpu, cuda=cuda, cpu_float64_reference=oracle,
                        gpu=torch.cuda.get_device_name(0), first_attempt=p['first_attempt'],
                        maximum_cross_device_gradient_error=max(float((first[key]-second[key]).abs().max()) for key in first),
                        shared_indices=[ids.tolist() for ids in draws], optimizer_steps=0, physical_substeps=0,
                        learning_updates=0, heldout_used=False, hardware_io=False))
    print('GETUP_CORRECTIVE_NATIVE_LOSS passed', torch.cuda.get_device_name(0), cpu, cuda, flush=True)


if __name__ == '__main__':
    main()
