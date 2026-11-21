"""Reuse the frozen lateral collector for a matched pulse and control in each direction."""

from unittest.mock import patch

import collect_roll_active_lateral as lateral
import roll_entry_snapshot as snapshot
from roll_head_support_control import HeadSupportRollPolicy, pulse_snapshot_slots


PULSE_PREFIX = 'pulse-'
LABELS = {
    **lateral.LABELS,
    'pulse-reference': '无扰动 · 头部脉冲',
    'pulse-left-push': '向左扰动 · 头部脉冲',
    'pulse-right-push': '向右扰动 · 头部脉冲',
}


def collect(env, obs, spec, record, endpoint, networks, homes, sim_home, limits, *, plan, page, check_active):
    roles = spec['perturbation']['roles']
    control_roles = ('reference', 'left-push', 'right-push')
    pulse_roles = tuple(PULSE_PREFIX + role for role in control_roles)
    assert tuple(roles) == (*control_roles, *pulse_roles)
    assert all(roles[role] == roles[PULSE_PREFIX + role] for role in control_roles)
    branch_order = iter(roles)
    write_velocity, original_slots = lateral.apply_roll_lateral_push, snapshot.roll_slots
    active_policy = None

    def policy(*args):
        nonlocal active_policy
        assert active_policy is None
        active_policy = HeadSupportRollPolicy(*args)
        return active_policy

    def slots(env, memory):
        assert memory is active_policy
        fields, generators = original_slots(env,memory)
        return pulse_snapshot_slots(fields,memory), generators

    def inject(env, eligible, heading, magnitude):
        # The collector restores the common state before this call.
        role = next(branch_order)
        assert magnitude == roles[role]
        active_policy.pulse_enabled = role.startswith(PULSE_PREFIX)
        return write_velocity(env,eligible,heading,magnitude)

    with patch.multiple(lateral, PLAN=plan, LABELS=LABELS, ProjectedRollPolicy=policy,
                        apply_roll_lateral_push=inject, create=True), patch.object(snapshot,'roll_slots',slots):
        lateral.collect(env,obs,spec,record,endpoint,networks,homes,sim_home,limits,page=page,
                        check_active=check_active)
    if (record/'evaluation.json').exists():
        assert next(branch_order,None) is None
