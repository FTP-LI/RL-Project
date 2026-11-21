"""Insert archived backward releases during native selective resets."""
import torch
from mjlab.managers.event_manager import EventTermCfg


class DifficultBackwardReleases:
    def __init__(self, env, slots, local_qpos, qvel):
        data = env.sim.data
        self.local_qpos = torch.as_tensor(local_qpos,device=env.device,dtype=data.qpos.dtype)
        self.qvel = torch.as_tensor(qvel,device=env.device,dtype=data.qvel.dtype)
        self.case_by_slot = torch.full((env.num_envs,),-1,device=env.device,dtype=torch.int64)
        slots = torch.as_tensor(slots,device=env.device,dtype=torch.int64)
        self.case_by_slot[slots] = torch.arange(len(slots),device=env.device)
        assert self.local_qpos.shape==(len(slots),data.qpos.shape[1])
        assert self.qvel.shape==(len(slots),data.qvel.shape[1])

    def apply(self, env, reset_ids):
        cases = self.case_by_slot[reset_ids]
        keep = cases>=0
        ids,case = reset_ids[keep],cases[keep]
        qpos = self.local_qpos[case]
        qpos[:,:3] += env.scene.env_origins[ids]
        # The native reset owns the manager resets and first observation
        # after this event. No forward or integration belongs here.
        env.sim.data.qpos[ids] = qpos
        env.sim.data.qvel[ids] = self.qvel[case]


def write_difficult_backward_release(env, env_ids):
    env.difficult_backward_releases.apply(env,env_ids)


def add_difficult_backward_event(cfg):
    name = 'difficult_backward_release'
    assert name not in cfg.events
    cfg.events[name] = EventTermCfg(func=write_difficult_backward_release,mode='reset')
