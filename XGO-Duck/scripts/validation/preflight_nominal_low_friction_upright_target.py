"""Check static head clearance and two-foot support in archived tilted states."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
import torch

from mjlab.scene import Scene
from preflight_nominal_timestep_low_friction import configured
from review_nominal_low_friction_roll_entry import first_mark, load_case
from review_nominal_timestep_friction_mechanisms import standing_states, standing_trace
from run_nominal_low_friction_roll_window import KEY as WINDOW, STAGE as PREVIOUS, parent, ROOT, PROJECT, read, rel, sha, write
from selective_friction_guard import STATIC_FIELDS, compare_static, resolve_foot_geometry_ids


KEY = 'nominal_low_friction_upright_target'
STAGE = 'nominal-low-friction-upright-target'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-upright-target'
VARIANTS = ('legs-only','legs-and-neck')


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'],project['staged_training_request']
    assert parent.allowed(project,PREVIOUS) and not task['simulation_running']
    assert not PLAN.exists() and not REPORT.exists() and KEY not in goal
    window = goal[WINDOW]
    observed = read(ROOT/window['observed_tilt_readback'],window['observed_tilt_readback_sha256'])
    assert observed['all_marker_checks_passed'] and not observed['reserved_seed_used']
    low = goal['nominal_timestep_low_friction_development']
    plant = read(ROOT/window['protocol'],window['protocol_sha256'])
    assert plant['composition']['foot_friction']==.8 and plant['environment']['seed']==1257
    assert plant['composition']['existing_low_friction_reserved_seed']==2246
    source = {}
    for node, fields in ((window,('protocol','collection','observed_tilt_readback','head_support_readback')),
                         (low,('protocol','collection'))):
        for name in fields:
            source[node[name]] = node[name+'_sha256']
    for file in (Path(__file__).resolve(),ROOT/plant['friction_development']['startup_fields']):
        source[rel(file)] = sha(file)
    for archive in observed['arms'].values():
        source[archive['observed_npz']] = archive['observed_npz_sha256']
    REPORT.mkdir()
    write(PLAN,dict(stage=STAGE,source=source,detector=observed['detector'],
        archived_low_friction={'earlier':low['collection'],'current':window['collection']},
        variants=list(VARIANTS),target_tilts_deg=[24.,12.],joint_delta_cap_rad=.25,
        translation_cap_m=.040,maximum_foot_displacement_m=.003,maximum_foot_rotation_deg=4.,
        maximum_foot_plane_distance_m=.003,minimum_head_gap_m=.002,minimum_support_margin_m=.002,
        support_mesh_vertex_band_m=.002,physical_steps=0,learning_updates=0,hardware_io=False,
        seed=1257,reserved_seed=2246))
    goal[KEY] = dict(status='frozen',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(PLAN))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Verify geometric recovery targets and support margin from the original archived plant.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_UPRIGHT_TARGET_FROZEN',sha(PLAN),flush=True)


def snapshots(node, role, detector, known=None):
    plan,folder,raw,poses = load_case(node)
    segment = raw['prefix'] if role is None else raw['branches'][role]
    folder /= 'shared' if role is None else role
    trace = standing_trace(folder,segment['chunks'],101,600)
    states = standing_states(plan,trace)
    if known is None:
        c = trace['control']
        lo,hi = detector['controls']
        window = (c>=lo)&(c<=hi)
        lo,hi = detector['signed_pitch_degrees']
        good = ((states['pitch']>=lo)&(states['pitch']<=hi)&
            (states['speed']<=detector['maximum_angular_speed_observation_norm'])&
            states['masks']['feet']&trace['head_contact'])
        hit = first_mark(c[window],good[window])
        known = {int(i):int(hit[i]) for i in np.flatnonzero(hit>=0)}
    assert set(known)==set(np.flatnonzero(~states['gate']))
    output = []
    for part in segment['chunks']:
        selected = [(world,k) for world,k in sorted(known.items()) if part['first']<=k<=part['last']]
        if not selected:
            continue
        path = folder/part['file']
        assert sha(path)==part['sha256']
        with np.load(path,allow_pickle=False) as item:
            poses_at_control = item['qpos']
        for world,k in selected:
            at = k-101
            assert trace['head_contact'][at,world] and trace['foot_contact'][at,world].all()
            output.append(dict(world=world,control=k,pose=str(poses[world]),chunk_sha256=part['sha256'],
                qpos=poses_at_control[k-part['first'],world].astype(np.float64)))
    assert len(output)==len(known)
    return output


class ContactGeometry:
    def __init__(self, model, source, policy_limits):
        self.model,self.data,self.source = model,mujoco.MjData(model),source
        self.feet = np.asarray(resolve_foot_geometry_ids(model))
        self.floor = int(np.flatnonzero(model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE).item())
        self.head = int(model.geom('robot/head_collision').id)
        assert model.geom_bodyid[self.head]==model.body('robot/jaw_soft').id
        assert all(int(model.geom_type[gid])==int(mujoco.mjtGeom.mjGEOM_MESH) for gid in (*self.feet,self.head))
        self.meshes = {}
        for gid in (*self.feet,self.head):
            mid = int(model.geom_dataid[gid])
            start, length = model.mesh_vertadr[mid],model.mesh_vertnum[mid]
            self.meshes[gid] = model.mesh_vert[start:start+length].astype(np.float64)
        root = model.body('robot/trunk_base').id
        mass_mask = model.body_rootid==root
        self.masses = np.where(mass_mask,model.body_mass,0.)
        assert self.masses.sum()>0
        names = [name for name in policy_limits if name.startswith(('left_','right_'))]+['neck_pitch']
        assert len(names)==11
        joints = [model.joint('robot/'+name) for name in names]
        self.names = names
        self.indices = np.asarray([item.qposadr.item() for item in joints])
        motor = np.asarray([policy_limits[name] for name in names])
        mechanical = np.asarray([item.range for item in joints])
        self.limits = np.column_stack((np.maximum(motor[:,0],mechanical[:,0]),
                                       np.minimum(motor[:,1],mechanical[:,1])))
        assert np.all(self.limits[:,0]<self.limits[:,1])

    def geometry(self, qpos):
        data = self.data
        data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model,data)
        rotation = data.geom_xmat.reshape(-1,3,3)
        np.testing.assert_allclose(rotation[self.floor][:,2],[0,0,1],rtol=0,atol=1e-10)
        plane = float(data.geom_xpos[self.floor,2])
        foot_vertices = [vertices@rotation[gid].T+data.geom_xpos[gid] for gid,vertices in self.meshes.items()
                         if gid!=self.head]
        head = self.meshes[self.head]@rotation[self.head].T+data.geom_xpos[self.head]
        center = (data.xipos*self.masses[:,None]).sum(axis=0)/self.masses.sum()
        foot_gaps = np.asarray([vertices[:,2].min()-plane for vertices in foot_vertices])
        gap = float(head[:,2].min()-plane)
        return dict(foot_positions=data.geom_xpos[self.feet].copy(),foot_frames=rotation[self.feet].copy(),
            foot_vertices=foot_vertices,foot_gaps=foot_gaps,head_gap=gap,com=center)

    def support(self, foot_vertices):
        band = self.source['support_mesh_vertex_band_m']
        ground = np.concatenate([vertices[vertices[:,2]<=vertices[:,2].min()+band,:2] for vertices in foot_vertices])
        try:
            return ConvexHull(ground).equations
        except QhullError:
            return None

    @staticmethod
    def margin(center, support):
        return None if support is None else float(-(support[:,:2]@center[:2]+support[:,2]).max())

    def fit(self, q, target_tilt, variant, base):
        source = self.source
        active = slice(None) if variant==VARIANTS[1] else slice(0,-1)
        coords,limits = self.indices[active],self.limits[active]
        cap, shift = source['joint_delta_cap_rad'],source['translation_cap_m']
        lb = np.r_[np.full(3,-shift),np.maximum(-cap,limits[:,0]-q[coords])]
        ub = np.r_[np.full(3,shift),np.minimum(cap,limits[:,1]-q[coords])]
        if not np.all(lb<ub):
            return dict(feasible=False,reason='no-joint-interval')
        initial = Rotation.from_quat(np.roll(q[3:7],-1))
        up = initial.as_matrix()[:,2]
        up_axis = np.cross(up,[0.,0.,1.])
        tilt = np.rad2deg(np.arctan2(np.linalg.norm(up_axis),up[2]))
        assert tilt>target_tilt and np.linalg.norm(up_axis)>.1
        oriented = q.copy()
        upright = Rotation.from_rotvec(up_axis/np.linalg.norm(up_axis)*np.deg2rad(tilt-target_tilt))*initial
        oriented[3:7] = np.roll(upright.as_quat(),1)
        base_polygon = self.support(base['foot_vertices'])
        if base_polygon is None:
            return dict(feasible=False,reason='no-two-foot-support-polygon')

        def geometry(delta):
            pose = oriented.copy()
            pose[:3] += delta[:3]
            pose[coords] += delta[3:]
            result = self.geometry(pose)
            relative = np.swapaxes(base['foot_frames'],-2,-1)@result['foot_frames']
            return (result,result['foot_positions']-base['foot_positions'],Rotation.from_matrix(relative).as_rotvec())

        def residual(delta):
            result, feet, rotation = geometry(delta)
            gap_error = max(0.,source['minimum_head_gap_m']+.0005-result['head_gap'])
            margin_error = max(0.,source['minimum_support_margin_m']+.0005-self.margin(result['com'],base_polygon))
            return np.r_[1000*feet.ravel(),30*rotation.ravel(),2500*gap_error,2000*margin_error,delta[3:]]

        fit = least_squares(residual,np.clip(np.zeros(len(lb)),lb,ub),bounds=(lb,ub),max_nfev=90)
        final, displacement, frame_error = geometry(fit.x)
        foot_motion = float(np.linalg.norm(displacement,axis=1).max())
        frame_motion = float(np.rad2deg(np.linalg.norm(frame_error,axis=1).max()))
        margin = self.margin(final['com'],self.support(final['foot_vertices']))
        feasible = (foot_motion<=source['maximum_foot_displacement_m'] and
            frame_motion<=source['maximum_foot_rotation_deg'] and
            abs(final['foot_gaps']).max()<=source['maximum_foot_plane_distance_m'] and
            final['head_gap']>=source['minimum_head_gap_m'] and
            margin is not None and margin>=source['minimum_support_margin_m'])
        assert abs(fit.x[3:]).max()<=cap+1e-9 and self.data.time==0.
        return dict(feasible=bool(feasible),solver_converged=bool(fit.success),target_tilt_deg=target_tilt,
            initial_tilt_deg=float(tilt),joint_names=self.names[active],joint_change_rad=fit.x[3:].tolist(),
            joint_target_rad=(q[coords]+fit.x[3:]).tolist(),maximum_foot_displacement_mm=foot_motion*1000,
            maximum_foot_rotation_deg=frame_motion,foot_plane_distance_mm=(1000*final['foot_gaps']).tolist(),
            head_clearance_mm=final['head_gap']*1000,com_support_margin_mm=None if margin is None else margin*1000)


def evaluate():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    node = goal[KEY]
    assert parent.allowed(project,STAGE) and not task['simulation_running'] and node['status']=='frozen'
    source = read(PLAN,node['protocol_sha256'])
    for file,digest in source['source'].items():
        assert sha(ROOT/file)==digest,file
    window = goal[WINDOW]
    plant = read(ROOT/window['protocol'],window['protocol_sha256'])
    cfg = configured(plant)
    model = Scene(cfg.scene,device='cpu').compile()
    with np.load(ROOT/plant['friction_development']['startup_fields'],allow_pickle=False) as prior:
        static = compare_static({key:np.asarray(getattr(model,key)) for key in STATIC_FIELDS},
                                {key:prior['static_'+key] for key in STATIC_FIELDS})
    limits = dict(zip(plant['walking']['joint_names'],plant['environment']['joint_limits'],strict=True))
    geom = ContactGeometry(model,source,limits)
    observed = read(ROOT/window['observed_tilt_readback'],window['observed_tilt_readback_sha256'])
    known = {int(k):v for k,v in observed['arms']['earlier-low-friction']['marked_controls'].items()}
    detector = source['detector']
    batches = dict(earlier=snapshots(goal['nominal_timestep_low_friction_development'],'lower-friction',detector,known),
                   current=snapshots(window,None,detector))
    assert [len(items) for items in batches.values()]==[14,15]
    summary = {}
    for kind, items in batches.items():
        for row in items:
            q = row.pop('qpos')
            base = geom.geometry(q)
            assert base['head_gap']<.002 and abs(base['foot_gaps']).max()<.003,(kind,row,base['foot_gaps'],base['head_gap'])
            margin = geom.margin(base['com'],geom.support(base['foot_vertices']))
            row.update(initial_head_clearance_mm=1000*base['head_gap'],
                initial_foot_plane_distance_mm=(1000*base['foot_gaps']).tolist(),
                initial_com_support_margin_mm=None if margin is None else 1000*margin,targets={})
            for tilt in source['target_tilts_deg']:
                row['targets'][str(tilt)] = {variant:geom.fit(q,tilt,variant,base) for variant in VARIANTS}
        summary[kind] = {str(tilt):{variant:sum(row['targets'][str(tilt)][variant]['feasible']
                                               for row in items) for variant in VARIANTS}
                         for tilt in source['target_tilts_deg']}
    assert geom.data.time==0. and not torch.cuda.is_initialized()
    output = REPORT/'geometry.json'
    assert not output.exists()
    write(output,dict(status='verified-saved-pose-feasibility',protocol_sha256=sha(PLAN),
        exact_static_model=static,counts=summary,batches=batches,physical_steps=0,
        estimated_foot_support_band_m=source['support_mesh_vertex_band_m'],
        dynamics_validated=False,learning_updates=0,hardware_io=False,reserved_seed_used=False))
    node.update(status='reviewed',evidence=rel(output),evidence_sha256=sha(output))
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=rel(output))
    task.update(info,next_gate='Select a joint recovery target using the geometric feasibility and gate it in paired dynamics.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_UPRIGHT_TARGET_REVIEWED',sha(output),summary,flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','evaluate'))
    action = cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'evaluate':evaluate}[action]()
