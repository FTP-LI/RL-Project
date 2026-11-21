"""Use archived poses to check a small pre-contact foot placement correction."""

from collections import Counter
from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np

from audit_nominal_supine_terminal_stance import attitude
from record_io import write
from review_nominal_skill_composition_handoff import read
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL
from run_nominal_skill_composition import PROJECT, ROOT, allowed, config, rel, sha


DESIGN = ROOT/'configs/experiments/nominal_supine_unloaded_foot_preflight.json'
KEY = 'nominal_supine_pre_bilateral_support'
SIDES = ('left', 'right')


class FootPlacement:
    def __init__(self, plant, settings):
        self.plant = plant
        self.data = mujoco.MjData(plant)
        self.spec = settings
        self.root = plant.body('robot/trunk_base').id
        self.floor, = np.flatnonzero(plant.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        self.head = plant.geom('robot/head_collision').id
        self.feet = [plant.geom(f'robot/{side}_foot_collision').id for side in SIDES]
        self.sites = [plant.site(f'robot/{side}_foot').id for side in SIDES]
        self.names = [[f'{side}_{joint}' for joint in ('hip_pitch', 'knee', 'ankle')] for side in SIDES]
        self.joints = [[plant.joint(f'robot/{name}').id for name in names] for names in self.names]
        self.jacp = np.empty((3, plant.nv))
        self.jacr = np.empty((3, plant.nv))

    def place(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_fwdPosition(self.plant, self.data)

    def floor_contacts(self):
        feet = np.zeros(2, dtype=bool)
        for contact in self.data.contact[:self.data.ncon]:
            if contact.efc_address >= 0 and self.floor in contact.geom:
                for i, foot in enumerate(self.feet):
                    feet[i] |= foot in contact.geom
        return feet

    def head_gaps(self):
        return np.array([mujoco.mj_geomDistance(self.plant, self.data, self.head, foot, .04, None)
                         for foot in self.feet])

    def estimate(self, qpos, observed_feet):
        s, plant, data = self.spec, self.plant, self.data
        _, roll, tilt = attitude(qpos[3:7])
        forward = np.r_[rotation(qpos[3:7])[:2, 0], 0.]
        heading_length = np.linalg.norm(forward)
        if (not s['minimum_tilt_deg'] <= tilt <= s['maximum_tilt_deg']
                or abs(roll) > s['maximum_absolute_roll_deg'] or heading_length < .3):
            return None, ['outside-pose-window']
        if observed_feet.all():
            return None, ['already-on-both-feet']
        forward /= heading_length
        lateral = np.array([-forward[1], forward[0], 0.])
        self.place(qpos)
        original_contacts = self.floor_contacts()
        if not np.array_equal(original_contacts, observed_feet):
            return None, ['contact-readback-mismatch']
        original_sites = data.site_xpos[self.sites].copy()
        old_penetration = np.minimum(self.head_gaps(), 0.)
        fore_gaps = (data.subtree_com[self.root] - original_sites) @ forward
        order = np.argsort(original_sites[:, 2], kind='stable')
        rejects = []
        for index in order:
            if observed_feet[index]:
                rejects.append('foot-already-loaded')
                continue
            self.place(qpos)
            site = self.sites[index]
            height = original_sites[index, 2] - data.geom_xpos[self.floor, 2]
            if height > s['maximum_foot_site_height_m']:
                rejects.append('foot-above-placement-window')
                continue
            gap = fore_gaps[index]
            if gap < s['minimum_fore_gap_from_com_m']:
                rejects.append('foot-not-behind-com')
                continue
            joints = self.joints[index]
            addresses = plant.jnt_qposadr[joints]
            columns = plant.jnt_dofadr[joints]
            mujoco.mj_jacSite(plant, data, self.jacp, self.jacr, site)
            linear = self.jacp[:, columns]
            angular = self.jacr[:, columns]
            # Hold sole pitch and site height to first order during the forward shift.
            system = np.vstack((forward @ linear, linear[2], .08 * (lateral @ angular)))
            requested = min(s['maximum_forward_motion_m'], gap)
            offset, _, rank, _ = np.linalg.lstsq(system, [requested, 0., 0.], rcond=None)
            if rank < 3:
                rejects.append('insufficient-sagittal-joint-authority')
                continue
            max_offset = float(np.max(np.abs(offset)))
            if max_offset > s['maximum_added_joint_offset_rad']:
                offset *= s['maximum_added_joint_offset_rad'] / max_offset
            trial = qpos.astype(float, copy=True)
            trial[addresses] += offset
            lower, upper = plant.jnt_range[joints].T
            if np.any((trial[addresses] < lower) | (trial[addresses] > upper)):
                rejects.append('joint-limit')
                continue
            self.place(trial)
            delta = data.site_xpos[site] - original_sites[index]
            travel, height_change, lateral_drift = delta @ forward, delta[2], delta @ lateral
            if (not s['minimum_forward_motion_m'] <= travel <= s['maximum_forward_motion_m']
                    or abs(height_change) > s['maximum_vertical_residual_m']
                    or abs(lateral_drift) > s['maximum_lateral_residual_m']):
                rejects.append('placement-travel-or-drift-limit')
                continue
            if not np.array_equal(self.floor_contacts(), original_contacts):
                rejects.append('changed-instantaneous-foot-contact')
                continue
            head_gaps = self.head_gaps()
            added_penetration = max(0., float(np.max(old_penetration - np.minimum(head_gaps, 0.))))
            if added_penetration > s['maximum_head_foot_penetration_increase_m']:
                rejects.append('increased-head-foot-penetration')
                continue
            return dict(foot=SIDES[index], joint_offsets_rad=dict(zip(self.names[index], offset.tolist(), strict=True)),
                        forward_mm=float(1000. * travel), foot_height_mm=float(1000. * height),
                        vertical_change_mm=float(1000. * height_change), lateral_change_mm=float(1000. * lateral_drift),
                        fore_com_gap_mm=float(1000. * gap), added_head_foot_penetration_mm=1000. * added_penetration), rejects
        return None, rejects


def chunk_at(folder, chunks, control, fields):
    chunk, = (row for row in chunks if row['first'] <= control <= row['last'])
    source = folder/chunk['file']
    assert sha(source) == chunk['sha256']
    row = control-chunk['first']
    with np.load(source, allow_pickle=False) as trace:
        assert trace['control'][row] == control
        fields_at_control = {key: trace[key][row] for key in fields}
    return fields_at_control, dict(path=rel(source), sha256=chunk['sha256'], completed_control=control)


def cohort(project, source, settings):
    registration = project['goal_execution'][source['key']]
    path = ROOT/registration['collection']
    record = read(path, source['collection_sha256'])
    protocol = read(ROOT/registration['protocol'], registration['protocol_sha256'])
    assert registration['status'] == 'reviewed' and record['protocol_sha256'] == registration['protocol_sha256']
    assert protocol['environment']['seed'] == source['seed'] not in settings['reserved_seeds']
    folder, series = path.parent, record
    control = settings['decision_control']
    pose_path = folder/'release-poses.npz'
    assert sha(pose_path) == record['release_sha256']
    with np.load(pose_path, allow_pickle=False) as released:
        worlds = np.flatnonzero(released['poses'] == 'supine')
    assert len(worlds) == record['actual_release_counts']['supine']
    if 'role' in source:
        series = record['branches'][source['role']]
    if 'prefix' in record and control <= record['prefix']['controls']:
        before, first_source = chunk_at(folder/'shared', record['prefix']['chunks'], control,
                                       ('qpos', 'foot_contact', 'active'))
    else:
        start = folder/source['role'] if 'role' in source else folder
        before, first_source = chunk_at(start, series['chunks'], control, ('qpos', 'foot_contact', 'active'))
    terminal = folder/source['role'] if 'role' in source else folder
    boundary = protocol['composition']['roll_start_control']
    assert boundary == 600 and protocol['roll_route']['standing_dwell_controls'] == 25
    assert protocol['roll_route']['roll']['criteria']['tilt_max_deg'] == 15
    after, final_source = chunk_at(terminal, series['chunks'], boundary, ('roll_phase',))
    passed = after['roll_phase'] == ROLL
    forward = set(protocol['seven_command']['groups']['forward'])
    planner = FootPlacement(Scene(config(protocol).scene, device='cpu').compile(), settings)
    results, rejection_counts = [], Counter()
    for world in worlds:
        if before['active'][world]:
            placement, rejected = planner.estimate(before['qpos'][world], before['foot_contact'][world])
        else:
            placement, rejected = None, ['inactive']
        rejection_counts.update(rejected)
        results.append(dict(world=int(world), original_roll_admitted=bool(passed[world]), is_forward=world in forward,
                            placement=placement, checks_rejected=rejected))
    no_admission = [row for row in results if not row['original_roll_admitted']]
    matched_forward = [row for row in no_admission if row['is_forward']]
    return dict(seed=source['seed'], collection_sha256=source['collection_sha256'],
                start_chunk=first_source, boundary_chunk=final_source, actual_supine=len(results),
                feasible=int(sum(row['placement'] is not None for row in results)),
                original_misses=len(no_admission), feasible_original_misses=sum(row['placement'] is not None for row in no_admission),
                forward_misses=len(matched_forward), feasible_forward_misses=sum(row['placement'] is not None for row in matched_forward),
                rejected_foot_checks=dict(rejection_counts),
                placements_and_original_misses=[row for row in results if row['placement'] is not None or not row['original_roll_admitted']])


def main():
    project, settings = read(PROJECT), read(DESIGN)
    assert allowed(project, settings['stage']) and not project['staged_training_request']['simulation_running']
    stage = project['goal_execution'][KEY]
    assert stage['status'] == 'archive-audit-reviewed'
    assert stage['analysis_sha256'] == settings['parent_screen_sha256']
    parent = ROOT/stage['analysis']
    screen = read(parent, settings['parent_screen_sha256'])
    assert screen['qualifies_observation_trigger'] is False
    early = project['goal_execution']['nominal_supine_early_support']
    design = read(ROOT/early['design'], early['design_sha256'])
    sources = [*design['sources'], dict(key='nominal_supine_early_support', seed=1257, role='original-getup',
                                       collection_sha256=screen['repeated_collection_sha256'])]
    tags = ('seed1256-archive', 'seed1257-archive', 'seed1257-same-seed-repeat')
    output = parent.with_name('unloaded-foot-preflight.json')
    assert not output.exists()
    cases = {}
    for tag, source in zip(tags, sources, strict=True):
        result = cohort(project, source, settings)
        cases[tag] = result
        print('UNLOADED_FOOT_GEOMETRY', tag, {k: v for k, v in result.items() if k not in ('placements_and_original_misses', 'start_chunk', 'boundary_chunk')}, flush=True)
    eligible = cases[tags[-1]]['feasible_forward_misses'] >= settings['minimum_forward_group_miss_coverage_in_latest_repeat']
    write(output, dict(protocol=rel(DESIGN), protocol_sha256=sha(DESIGN), script_sha256=sha(Path(__file__)),
                       parent_screen_sha256=sha(parent), all_cohorts=cases, geometry_supports_matched_probe=eligible,
                       new_physics_steps=0, learning_updates=0, reserved_seeds_used=False, hardware_io=False))
    stage.update(unloaded_foot_preflight=rel(output), unloaded_foot_preflight_sha256=sha(output),
                 unloaded_foot_geometry_supports_matched_probe=eligible)
    project['staged_training_request']['current_evidence'] = project['current_evidence'] = rel(output)
    write(PROJECT, project)
    print('UNLOADED_FOOT_PREFLIGHT_COMPLETE', sha(output), eligible, flush=True)


if __name__ == '__main__':
    main()
