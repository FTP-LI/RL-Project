"""First-episode accounting shared by fixed-policy evaluations.

Failed episodes retain their own metrics and never contribute after a reset.
Timeout is success only at the requested horizon with no failure term. A fall
on the final step is still a failure. Error means give each episode equal weight;
the accompanying duration/count prevents short failed episodes being hidden.
"""
import numpy as np


class FirstEpisodeMetrics:
    def __init__(self, initial_xy, horizon_steps, dt):
        self.initial_xy = np.asarray(initial_xy, dtype=float).copy()
        self.n = len(initial_xy)
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.active = np.ones(self.n, dtype=bool)
        self.count = np.zeros(self.n, dtype=int)
        self.xy_sum = np.zeros(self.n)
        self.yaw_sum = np.zeros(self.n)
        self.delta_sq = np.zeros(self.n)
        self.slip_sum = np.zeros(self.n)
        self.contacts = np.zeros(self.n, dtype=int)
        self.drift = np.zeros(self.n)
        self.reason = ['incomplete'] * self.n

    def update(self, step, xy, velocity, yaw_rate, command, action_delta,
               foot_speed, contact, terminated, truncated, fell, nan, out_of_bounds):
        active = self.active.copy()
        self.count[active] += 1
        # Use the planar vector error, so a stationary robot does not pass the
        # 0.15 m/s forward command merely by averaging its error with zero y.
        self.xy_sum[active] += np.linalg.norm(velocity[active] - command[:2], axis=1)
        self.yaw_sum[active] += np.abs(yaw_rate[active] - command[2])
        self.delta_sq[active] += np.square(action_delta[active]).mean(axis=1)
        self.slip_sum[active] += (foot_speed[active] * contact[active]).sum(axis=1)
        self.contacts[active] += contact[active].sum(axis=1).astype(int)
        self.drift[active] = np.linalg.norm(xy[active] - self.initial_xy[active], axis=1)
        done = terminated | truncated | fell | nan | out_of_bounds
        for i in np.flatnonzero(active & (done | (step >= self.horizon_steps))):
            if nan[i]:
                reason = 'nan_state'
            elif fell[i]:
                reason = 'fell_over'
            elif out_of_bounds[i]:
                reason = 'out_of_terrain_bounds'
            elif terminated[i]:
                reason = 'other_termination'
            elif step < self.horizon_steps:
                reason = 'early_timeout'
            else:
                reason = 'horizon_complete'
            self.reason[i] = reason
            self.active[i] = False
        return active

    def result(self):
        rows = []
        for i in range(self.n):
            count = int(self.count[i])
            rows.append(dict(
                env_index=i, outcome=self.reason[i], steps=count,
                survival_seconds=count*self.dt,
                linear_xy_mae_m_s=float(self.xy_sum[i]/count) if count else None,
                yaw_mae_rad_s=float(self.yaw_sum[i]/count) if count else None,
                action_delta_rms=float(np.sqrt(self.delta_sq[i]/count)) if count else None,
                contact_foot_slip_m_s=float(self.slip_sum[i]/self.contacts[i]) if self.contacts[i] else None,
                contact_samples=int(self.contacts[i]), final_displacement_m=float(self.drift[i])))
        def mean(name):
            values = [r[name] for r in rows if r[name] is not None]
            return float(np.mean(values)) if values else None
        return dict(
            episodes=rows,
            full_episode_survival=float(np.mean([r == 'horizon_complete' for r in self.reason])),
            fall_count_excluding_timeouts=self.reason.count('fell_over'),
            outcome_counts={name: self.reason.count(name) for name in sorted(set(self.reason))},
            mean_survival_seconds=mean('survival_seconds'),
            linear_xy_mae_m_s=mean('linear_xy_mae_m_s'), yaw_mae_rad_s=mean('yaw_mae_rad_s'),
            action_delta_rms=mean('action_delta_rms'),
            contact_foot_slip_m_s=mean('contact_foot_slip_m_s'),
            episodes_without_contact=int((self.contacts == 0).sum()),
            final_displacement_m_median=float(np.median(self.drift)))
