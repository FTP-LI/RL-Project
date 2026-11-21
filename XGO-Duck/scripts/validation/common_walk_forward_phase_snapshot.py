"""Reuse the full scene inventory with the pinned stateless upright reward."""
from types import SimpleNamespace

from mjlab.tasks.velocity.mdp.rewards import upright

from getup_release_fork_state import digest, signature
from getup_release_scene_state import SceneReleaseSnapshot


class WalkingSnapshot(SceneReleaseSnapshot):
    def __init__(self, env, limit):
        manager = env.reward_manager
        terms = manager._class_term_cfgs
        assert len(terms) == 1 and not env.metrics_manager._class_term_cfgs
        term = terms[0].func
        assert type(term) is upright and term._env is env and term._terrain_sensor_names is None
        assert set(vars(term)) == {'_terrain_sensor_names', '_debug_vis_enabled', '_env', '_asset_cfg'}
        self.upright_config = terms[0]
        self.upright_sha = self.upright_signature()

        # The pinned upright callback and reset do not write any class fields.
        # Expose that validated case to the existing inventory while binding
        # its mutable manager accumulators to the real manager for restore.
        reward_view = SimpleNamespace(**vars(manager))
        reward_view._class_term_cfgs = ()
        view = SimpleNamespace(**vars(env))
        view.device = env.device
        view.reward_manager = reward_view
        super().__init__(view, SimpleNamespace(), limit)
        owners = {id(view): env, id(reward_view): manager}
        self.slots = [(label, owners.get(id(owner), owner), field) for label, owner, field in self.slots]
        self.env = env

    def upright_signature(self):
        cfg = self.upright_config
        term = cfg.func
        return digest(signature(dict(parameters=cfg.params, weight=cfg.weight, asset=term._asset_cfg,
                                     terrain=term._terrain_sensor_names, debug=term._debug_vis_enabled)))

    def verify(self):
        assert self.upright_signature() == self.upright_sha
        return dict(super().verify(), stateless_upright_config_sha256=self.upright_sha)
