"""Reconstruct sagittal support margins at events recorded in the matched trial."""

from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np

from audit_nominal_supine_terminal_stance import attitude
from roll_metrics import rotation
from run_nominal_skill_composition import config
from run_nominal_supine_terminal_stance import KEY, PROJECT, REPORT, ROOT, read, sha, write
from verify_nominal_skill_composition import read_trace


FIELDS = ('qpos','qvel','foot_contact')


def first(controls, mask):
    selected = controls[mask]
    return None if not len(selected) else int(selected[0])


def sagittal_support(model, data, heading, root_body, floor, feet):
    forward_centroid = data.subtree_com[root_body,:2] @ heading
    contact_groups = {foot:[] for foot in feet}
    for i in range(data.ncon):
        contact = data.contact[i]
        a,b = map(int,contact.geom)
        foot = b if a == floor else a if b == floor else -1
        if foot in contact_groups and contact.efc_address >= 0:
            contact_groups[foot].append(float(contact.pos[:2] @ heading))
    contacts = [x for positions in contact_groups.values() for x in positions]
    if not contacts:
        return dict(both_feet=False,contact_points=0,front_margin_mm=None,rear_margin_mm=None)
    return dict(both_feet=all(contact_groups.values()),contact_points=len(contacts),
        front_margin_mm=float((max(contacts)-forward_centroid)*1000),
        rear_margin_mm=float((forward_centroid-min(contacts))*1000))


def main():
    state = read(PROJECT)['goal_execution'][KEY]
    plan = read(ROOT/state['protocol'],state['protocol_sha256'])
    checked = read(ROOT/state['pair_analysis'],state['pair_analysis_sha256'])
    collection = ROOT/state['collection']
    raw = read(collection,state['collection_sha256'])
    assert checked['original_pre_roll_gate_verified']
    spec = plan['terminal_stance']
    start,end = spec['fork_control'],spec['end_control']
    folder = collection.parent
    shared = read_trace(folder/'shared',raw['prefix']['chunks'],1,start,FIELDS,sha)
    worlds = np.asarray(checked['eligible_worlds'])
    headings = rotation(shared['qpos'][-1,worlds,3:7])[:,:2,0]
    headings /= np.linalg.norm(headings,axis=-1,keepdims=True)
    model = Scene(config(plan).scene,device='cpu').compile()
    mj = mujoco.MjData(model)
    floor, = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    feet = [model.geom(f'robot/{side}_foot_collision').id for side in ('left','right')]
    root_body = model.body('robot/trunk_base').id
    neck = plan['walking']['joint_names'].index('neck_pitch')

    def read_frame(record, k, world, heading):
        source, index = (shared,k-1) if k <= start else (record,k-start-1)
        mujoco.mj_resetData(model,mj)
        mj.qpos[:] = source['qpos'][index,world]
        mj.qvel[:] = source['qvel'][index,world]
        mujoco.mj_forward(model,mj)
        support = sagittal_support(model,mj,heading,root_body,floor,feet)
        archived = bool(source['foot_contact'][index,world].all())
        support.update(control=k,archived_dual_foot_contact=archived,
            cpu_dual_foot_matches_archived=archived == support['both_feet'],
            root_pitch_roll_tilt_deg=attitude(mj.qpos[3:7]).tolist())
        return support

    results = {}
    for arm in spec['roles']:
        branch = raw['branches'][arm]
        prefix = folder/arm
        contact_data = prefix/'terminal-stance.npz'
        assert sha(contact_data) == branch['terminal_stance_sha256']
        with np.load(contact_data,allow_pickle=False) as saved:
            control = saved['control']
            head = saved['head_foot_contact']
            feet_recorded = saved['dual_foot_contact']
            # The neck is the only overridden coordinate before the stance stage.
            overrides = saved['target_overridden'].copy()
            overrides[:,:,neck] = False
            leg = overrides.any(axis=-1)
            pitch = saved['pitch_deg']
        trace = read_trace(prefix,branch['chunks'],start+1,end,FIELDS,sha)
        np.testing.assert_array_equal(feet_recorded,trace['foot_contact'][:,worlds].all(axis=-1))
        rows = []
        for i,world in enumerate(worlds):
            separation = first(control,~head[:,i])
            legs = first(control,leg[:,i])
            events = dict(fork=start,before_first_separation=None if separation is None else separation-1,
                first_separation=separation,first_double_support_loss=first(control,~feet_recorded[:,i]),
                first_leg_target=legs,before_first_leg_target=None if legs is None else legs-1,
                first_pitch_over_45_deg=first(control,pitch[:,i]>45.),
                terminal=end)
            rows.append(dict(world=int(world),events={name:None if k is None else read_frame(trace,k,world,headings[i])
                for name,k in events.items()}))
        results[arm] = rows
    output = REPORT/'support-chronology.json'
    assert not output.exists()
    write(output,dict(protocol_sha256=state['protocol_sha256'],pair_analysis_sha256=state['pair_analysis_sha256'],
        collection_sha256=state['collection_sha256'],script_sha256=sha(Path(__file__)),seed=checked['seed'],
        event_scope='Live CUDA head, foot and target data. CPU re-evaluates saved ground contact-point geometry and whole-body COM.',
        positive_front_margin_means_com_behind_front_contact=True, generated_physical_steps=0, arms=results))
    print('SUPINE_SUPPORT_CHRONOLOGY',sha(output),flush=True)


if __name__ == '__main__':
    main()
