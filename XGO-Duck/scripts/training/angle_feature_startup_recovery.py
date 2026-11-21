"""Zero-step startup guard repair; frozen failed source remains intact.

The official training task samples a3-component command at48..50. Only
head51..54 and body padding56..60 are required zero; angle55 is masked solely
for the source reference. This changes a startup assertion, not model inputs,
physical configuration, rewards, initialization, learning or acceptance.
The explicit runner below preserves all other frozen checks and recordings.
"""
from angle_feature_pilot import *

def assert_startup_padding(reference_obs):
    np.testing.assert_allclose(reference_obs[:,51:], 0, atol=1e-7, rtol=0)

def runner_class(base, plan, record, metadata):
    parent = core.runner_class(base, plan, record, metadata)

    class AnglePilot(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sample = self.video.sample

            def observed_video():
                step = int(self.env.unwrapped.common_step_counter)
                if step == 700:
                    assert self.video.writer is None
                    self.video.last_started = None
                    write(record/'learned-clip-trigger.json', dict(
                        common_step=step, completed_updates=self.completed,
                        actor_updates_completed=max(0, self.completed-14), extra_physics_steps=0))
                sample()
            self.video.sample = observed_video

        def _live_check(self, imported):
            import onnxruntime as ort
            from tensordict import TensorDict
            env = self.env.unwrapped
            contract = neutralize_slot(self.alg.actor)
            if plan['arm'] == 'control':
                self.angle_hook = self.alg.actor.mlp[0].weight.register_hook(mask_angle_gradient)
            # Parent records the imported state before this dynamic override.
            # The warmup invariant must instead use the reparameterized state.
            self.initial_actor = {k:v.detach().cpu().clone() for k,v in self.alg.actor.state_dict().items()}
            names = metadata['joint_names'].split(',')
            action = env.action_manager.get_term('joint_pos')
            assert list(action.target_names) == names
            home = np.asarray(metadata['default_joint_pos'].split(','), np.float32)
            offset = action._offset.detach().cpu().numpy()
            np.testing.assert_allclose(offset, np.broadcast_to(home, offset.shape), atol=1e-7, rtol=0)
            obs = self.env.get_observations()[:32]
            actual_obs = obs['actor'].detach().cpu().numpy().copy()
            reference_obs = actual_obs.copy()
            reference_obs[:, ANGLE_SLOT] = 0
            np.testing.assert_allclose(actual_obs[:,34:48], 0, atol=1e-7, rtol=0)
            assert_startup_padding(reference_obs)
            ids, _ = env.scene['robot'].find_joints(names, preserve_order=True)
            default = env.scene['robot'].data.default_joint_pos[:,ids].cpu().numpy()
            np.testing.assert_allclose(default, np.broadcast_to(plan['simulation_home'], default.shape), atol=1e-7, rtol=0)
            with torch.no_grad(): actual = self.alg.actor(obs).detach().cpu().numpy()
            options = ort.SessionOptions(); options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(ROOT/plan['model']), sess_options=options, providers=['CPUExecutionProvider'])
            expected = np.concatenate([session.run(None, {'obs':x[None]})[0] for x in reference_obs])
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            assert env.common_step_counter == 0
            np.savez_compressed(record/'live-import-parity.npz', observations=actual_obs,
                original_reference_observations=reference_obs, critic_observations=obs['critic'].cpu().numpy(),
                torch_actions=actual, onnx_actions=expected,
                initial_qpos=env.sim.data.qpos.detach().cpu().numpy(), initial_qvel=env.sim.data.qvel.detach().cpu().numpy())
            write(record/'live-import-parity.json', dict(status='passed', maximum_error=float(abs(actual-expected).max()),
                physical_steps=0, action_home_matches=True, source_reset_home_matches=True,
                initial_raw_history_zero=True, imported=imported, angle_contract=contract,
                observations='Actual noisy angle-task inputs; source reference masks only slot55 to0'))

            prior = read(ROOT/'records/reports/roll-action-memory/startup-recovery/control/progress.json')
            source = Path(prior['path'])/'forward-roll-trace.npz'
            assert sha(source) == plan['drift_probe_trace_sha256']
            with np.load(source) as saved:
                enriched = saved['observations'].copy()
                enriched[:,:,ANGLE_SLOT] = gravity_phase(enriched[:,:,3:6]).astype(np.float32)
                indices = [1,16,31,46,61,76,151,301,450]
                probe = enriched[indices].reshape(-1,61).copy()
                target = saved['raw_policy_actions'][indices].reshape(-1,14).copy()
            np.savez_compressed(record/'phase-drift-probe.npz', observations=probe, source_actions=target)
            self.probe = TensorDict({'actor':torch.as_tensor(probe, device=env.device)}, batch_size=[len(probe)])
            self.teacher_probe = torch.as_tensor(target, device=env.device)
            with torch.no_grad(): initial_probe = self.alg.actor(self.probe)
            torch.testing.assert_close(initial_probe, self.teacher_probe, atol=2e-5, rtol=2e-5)

            fields = {k:getattr(env.sim.model,k).detach().cpu().numpy().copy() for k in sorted(env.sim.expanded_fields)}
            np.savez_compressed(record/'training-expanded-initial.npz', **fields)
            comparison = []
            if plan['arm'] == 'candidate':
                other = ROOT/'records/experiments'/plan['matching_control_run']
                with np.load(other/'live-import-parity.npz') as old:
                    for key, value in (('observations', actual_obs), ('critic_observations', obs['critic'].cpu().numpy()),
                        ('initial_qpos', env.sim.data.qpos.cpu().numpy()), ('initial_qvel', env.sim.data.qvel.cpu().numpy())):
                        np.testing.assert_array_equal(value, old[key])
                with np.load(other/'training-expanded-initial.npz') as old:
                    comparison = compare_fields(fields, {k:old[k].copy() for k in old.files})
                saved = torch.load(Path(read(other/'runner.json')['log_dir'])/'model_init.pt', map_location='cpu', weights_only=True)
                for role, network in (('actor', self.alg.actor), ('critic', self.alg.critic)):
                    assert all(torch.equal(v.detach().cpu(),saved[role+'_state_dict'][k]) for k,v in network.state_dict().items())
                write(record/'matched-start.json', dict(status='passed', physical_steps=0, initial_actor_critic_exact=True,
                    initial_observations_and_physical_state_match=True, control_run=plan['matching_control_run']))
            write(record/'angle-input-contract.json', dict(status='passed', physical_steps=0, arm=plan['arm'],
                actor_angle_access=plan['arm']=='candidate', common_critic_receives_phase=True,
                initial_column_exact_zero=True, normalizer=contract, same_time_field_comparison=comparison,
                expanded_initial_sha256=sha(record/'training-expanded-initial.npz'),
                drift_probe_sha256=sha(record/'phase-drift-probe.npz')))

        def _publish(self, status, error=None):
            if plan['arm'] == 'control':
                assert not torch.count_nonzero(self.alg.actor.mlp[0].weight[:,ANGLE_SLOT]).item()
            super()._publish(status, error)

        def _export(self):
            super()._export()
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx
            folder = ROOT/'artifacts'/plan['run_id']
            contract = dict(slot=ANGLE_SLOT, units='signed cumulative radians', source='unwrapped observed projected gravity',
                zero_at_roll_trigger=True, actor_angle_access=plan['arm']=='candidate',
                hardware_estimator_validated=False, other_inputs='Original61 layout; slot56..60 zero')
            attach_metadata_to_onnx(str(folder/'policy.onnx'), {'roll_angle_contract':__import__('json').dumps(contract,sort_keys=True)})
            write(folder/'input-contract.json', contract)
            manifest = read(folder/'manifest.json')
            manifest.update(model_sha256=sha(folder/'policy.onnx'), angle_input_contract=contract)
            write(folder/'manifest.json', manifest)
    return AnglePilot

def run(planfile):
    planfile = Path(planfile); plan = read(planfile)
    assert os.isatty(1), 'Use the one visible VS Code task terminal'
    assert read(ROOT/'project.json')['staged_training_request']['current_stage'] == 'S4-angle-feature-startup-recovery'
    for name,digest in plan['source_sha256'].items(): assert sha(ROOT/name)==digest, name
    for name,digest in plan['training_source_sha256'].items(): assert sha(core.WORK/name)==digest, name
    for name,digest in plan['environment_source_sha256'].items(): assert sha(Path(name))==digest, name
    cfg, display, metadata = prepare(plan)
    record = ROOT/'records/experiments'/plan['run_id']; record.mkdir(exist_ok=False)
    write(record/'protocol.json',plan); write(record/'display-state.json',display)
    source_dir = record/'sources'; source_dir.mkdir()
    for name in plan['source_sha256']: shutil.copy2(ROOT/name,source_dir/Path(name).name)
    write(ROOT/'records/training-current.json',dict(record=str(record),protocol=str(planfile)))
    write(record/'run-event.json',dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        foreground_tty=os.isatty(1),protocol=str(planfile),protocol_sha256=sha(planfile)))
    write(record/'state.json',dict(status='starting',run_id=plan['run_id'],completed_updates=0,clips=[]))
    from evaluate_standup_reference import Tee
    with (record/'training.log').open('a',buffering=1) as output:
        oldout,olderr=sys.stdout,sys.stderr; sys.stdout,sys.stderr=Tee(oldout,output),Tee(olderr,output)
        try:
            from mjlab.scripts import train
            base=train.load_runner_cls(plan['task']); train.load_runner_cls=lambda task:runner_class(base,plan,record,metadata)
            train.launch_training(plan['task'],cfg)
        except BaseException as error:
            state=read(record/'state.json'); state.update(status='stopped-review-required',error=repr(error))
            write(record/'state.json',state); raise
        finally: sys.stdout,sys.stderr=oldout,olderr

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('protocol');run(p.parse_args().protocol)
