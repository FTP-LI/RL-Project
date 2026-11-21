"""Register the right-side term in the actual plant and check its CPU boundary."""
from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from common_scene_contract import compare_arrays, model_arrays
from getup_early_support_reward import inputs
from getup_right_upright_progress import RightUprightProgress
from getup_teacher_anchor_train import prepare


TERM = 'right_side_upright_progress'
DRAFT = ROOT/'configs/experiments/getup_right_side_upright_progress_draft.json'
PARENT = ROOT/'configs/experiments/getup_early_support_clock.json'
SCENE = ROOT/'records/reports/getup-interface-value/cpu-scene-fields.npz'
OUT = ROOT/'records/reports/getup-upright-recovery-pair'
DEST = OUT/'right-side-upright-native-cpu.json'
COMPONENT = OUT/'right-side-upright-reward-cpu.json'


def native_reward(env):
    # mjlab has computed termination and native final-substep contacts at
    # this boundary. The actor sees the refreshed observation only afterward.
    termination = env.termination_manager
    active = ~(termination.terminated|termination.time_outs)
    return env._right_upright_progress.advance(*inputs(env),env.episode_length_buf,active)


def reset_reward(env,env_ids):
    component = getattr(env,'_right_upright_progress',None)
    if component is not None:
        component.reset(slice(None) if env_ids is None else env_ids)


def configuration(environment):
    from mjlab.managers import EventTermCfg, RewardTermCfg

    cfg,display,_ = prepare(environment)
    assert cfg.env.auto_reset and cfg.env.episode_length_s==8. and cfg.env.decimation==5
    assert TERM not in cfg.env.rewards and TERM not in cfg.env.events
    cfg.env.rewards[TERM] = RewardTermCfg(func=native_reward,weight=1.)
    cfg.env.events[TERM] = EventTermCfg(func=reset_reward,mode='reset')
    return cfg,display


def install_state(env,draft,release_age,alignment):
    source = read(ROOT/draft['trigger_transfer'])
    state = RightUprightProgress(env.num_envs,env.device,release_age,alignment,source['reused_entry'],
                                 draft['proposed_reward_window_controls'])
    env._right_upright_progress = state
    return state


def main():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import RewardManager

    assert not DEST.exists() and not torch.cuda.is_initialized()
    draft,parent,proof = read(DRAFT),read(PARENT),read(COMPONENT)
    assert draft['status']=='draft-not-executable' and proof['status']=='passed-cpu-reward-state-replay'
    assert not draft['heldout_used'] and not proof['physical_substeps']
    for path,digest in proof['source_sha256'].items():
        assert sha(ROOT/path)==digest,path
    step = inspect.getsource(ManagerBasedRlEnv.step)
    ordered = ('self.episode_length_buf += 1','self.termination_manager.compute()',
               'self.reward_manager.compute(','self._reset_idx(reset_env_ids)','self.sim.forward()',
               'self.obs_buf = self.observation_manager.compute(')
    offsets = [step.index(token) for token in ordered]
    assert offsets==sorted(set(offsets))
    installed_compute = RewardManager.compute
    wrapper_source = inspect.getsource(installed_compute)
    assert '_orig_reward_compute(self, dt)' in wrapper_source and 'torch.nan_to_num(result' in wrapper_source
    manager_compute = installed_compute.__globals__['_orig_reward_compute']
    manager_source = inspect.getsource(manager_compute)
    assert 'term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * scale' in manager_source
    assert 'self._step_reward[:, term_idx] = value / scale' in manager_source
    library = {str(file):sha(file) for file in map(Path,(
        inspect.getsourcefile(ManagerBasedRlEnv),inspect.getsourcefile(manager_compute),inspect.getsourcefile(installed_compute)))}
    env_spec = deepcopy(parent['environment'])
    env_spec.update(num_envs=16,seed=1098)
    env_spec['split']['pose_codes'] = [0,1,2,3]*4
    torch.set_num_threads(4)
    cfg,_ = configuration(env_spec)
    env = ManagerBasedRlEnv(cfg=cfg.env,device='cpu',render_mode=None)
    try:
        with np.load(SCENE) as stored:
            compare_arrays(model_arrays(env.sim.mj_model),{name:stored[name] for name in stored.files})
        state = install_state(env,draft,parent['release_age'],parent['alignment'])
        obs,_ = env.reset(seed=env_spec['seed'])
        assert env.reward_manager._term_names[-1]==TERM and env.reward_manager._scale_by_dt
        assert 'reset' in env.event_manager.available_modes
        torch.testing.assert_close(native_reward(env),torch.zeros(env.num_envs),rtol=0,atol=0)
        synthetic = obs['actor'].clone()
        synthetic[:3,3:6] = torch.tensor([[0.,-1.,0.],[0.,-1.,0.],[0.,-1.,0.]])
        ages = torch.zeros_like(env.episode_length_buf)
        ages[:3] = parent['release_age']
        state.observe_release(synthetic,ages)
        assert state.right[:3].all()
        state.trigger_age[:3] = 120
        state.supported[:,:3] = True
        state.previous_potential[:3] = .5
        ids = torch.tensor([0,2],dtype=torch.long)
        env.reset(env_ids=ids)
        assert not state.release_known[ids].any() and not state.right[ids].any()
        assert (state.trigger_age[ids]==-1).all() and not state.supported[:,ids].any()
        assert not state.previous_potential[ids].any()
        assert state.release_known[1] and state.right[1] and state.trigger_age[1]==120
        assert state.supported[:,1].all() and state.previous_potential[1]==.5
        assert env.common_step_counter==env._sim_step_counter==0
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    files = (DRAFT,PARENT,SCENE,COMPONENT,Path(__file__),ROOT/'scripts/training/getup_right_upright_progress.py',
             ROOT/'scripts/training/getup_early_support_reward.py')
    write(DEST,dict(status='passed-native-cpu-registration',physical_substeps=0,learning_updates=0,
        heldout_used=False,worlds=16,release_observation_size=61,selective_reset_ids=[0,2],
        kept_synthetic_state_world=1,other_reward_terms_preserved=True,reward_order=list(ordered),
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in files},library_sha256=library))
    print('RIGHT_UPRIGHT_NATIVE_CPU',16,ids.tolist(),'controls=0',flush=True)


if __name__=='__main__':
    main()
