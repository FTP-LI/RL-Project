"""Read geometry and whole-body motion before first bilateral support."""

from argparse import ArgumentParser
import fcntl
from pathlib import Path

import numpy as np

import review_nominal_low_friction_early_transfer_authority as authority
import review_nominal_low_friction_forward_support as measured
import replay_nominal_low_friction_forward_support as previous
from roll_metrics import rotation


experiment = measured.experiment
ROOT, PROJECT, read, rel, sha, write = experiment.ROOT, experiment.PROJECT, measured.read, measured.rel, measured.sha, measured.write
KEY, STAGE = 'nominal_low_friction_forward_support_geometry', 'nominal-low-friction-forward-support-geometry'
PLAN = experiment.PLAN.with_stem(KEY)
REPORT = experiment.REPORT/'pre-support-geometry'
START, END = 110, 160
FROZEN_WINDOWS = (('release-480',480),('release-560',560),('release-640',640))
OFFSETS = (-160,-120,-80,-40,0)
FIELDS = ('qpos','qvel','foot_contact')
METRICS = ('tilt_deg','upright_rate_deg_s','fore_angular_momentum_kgm2_s','forward_com_velocity_mm_s',
    'com_foot_midpoint_mm','left_site_forward_mm','right_site_forward_mm','left_clearance_mm','right_clearance_mm',
    'front_contact_margin_mm','rear_contact_margin_mm')


def freeze():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert experiment.parent.allowed(project,previous.STAGE) and not task['simulation_running']
    assert KEY not in goal and not PLAN.exists() and not REPORT.exists()
    capture, replay, historical, setup = (goal[k] for k in (experiment.KEY,previous.KEY,
        'nominal_low_friction_support_effort_limit_readback',authority.KEY))
    assert all(x['status']=='reviewed' for x in (capture,replay,historical,setup))
    design = read(ROOT/capture['protocol'],capture['protocol_sha256'])
    assert (design['environment']['seed'],design['composition']['foot_friction'])==(1257,.8)
    assert design['forward_support']['heldout_seed']==2246 and design['roll_route']['standing_dwell_controls']==25
    assert set(design['seven_command']['minimum_by_command'].values())=={122} and design['seven_command']['minimum_overall']==852
    assert not any(x['eligible_for_observable_preflight'] for x in read(ROOT/replay['result'],replay['result_sha256'])['summary'].values())
    original = read(ROOT/historical['result'],historical['result_sha256'])['release_support']['matches']
    pairs = {str(row['missed']['world']):row['admitted']['world'] for row in original}
    worlds = sorted({int(w) for w in pairs}|set(pairs.values()))
    assert len(pairs)==13 and len(worlds)==26 and set(worlds)<=set(design['forward_support']['selected_worlds'])
    sources = {row[key]:row[key+'_sha256'] for row,keys in ((capture,('protocol','collection','result')),
        (replay,('result',)),(historical,('result',)),(setup,('protocol',))) for key in keys}
    for path in (__file__,authority.__file__,measured.__file__):
        path = Path(path).resolve()
        sources[rel(path)] = sha(path)
    for name,digest in sources.items():
        assert sha(ROOT/name)==digest,name
    record = dict(stage=STAGE,seed=1257,friction=.8,controls=[START,END],step_ms=20,
        original_release_control=100,fixed_release_offsets_ms=dict(FROZEN_WINDOWS),bilateral_load_aligned_offsets_ms=list(OFFSETS),
        earlier_capture_protocol=capture['protocol'],earlier_capture_result=capture['result'],
        earlier_capture_collection=capture['collection'],geometry_authority_protocol=setup['protocol'],
        sources=sources,worlds=worlds,archive_pairs=pairs,focus=[439,779],roles=list(measured.ROLES),
        contact_parity_minimum=.95,heldout_seed=2246,new_physical_steps=0,learning_updates=0,hardware_io=False)
    REPORT.mkdir()
    write(PLAN,record)
    goal[KEY] = dict(status='archive-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),hardware_io=False)
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-frozen',PLAN,
        'Reconstruct pre-support geometry and compare fixed and actual load-aligned timing.')
    write(PROJECT,project)
    print('FORWARD_GEOMETRY_FROZEN',sha(PLAN),len(worlds),flush=True)


class Geometry:
    def __init__(self, engine):
        self.engine = engine
        self.model,self.mj,self.data = engine.model,engine.mj,engine.data
        model = self.model
        self.root = model.body('robot/trunk_base').id
        self.feet = [model.geom(f'robot/{side}_foot_collision').id for side in ('left','right')]
        self.sites = [model.site(f'robot/{side}_foot').id for side in ('left','right')]
        self.vertices = []
        for foot in self.feet:
            assert model.geom_type[foot]==self.mj.mjtGeom.mjGEOM_MESH
            mesh = model.geom_dataid[foot]
            offset,count = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
            self.vertices.append(model.mesh_vert[offset:offset+count].astype(np.float64))

    def select(self, world):
        for name,values in self.engine.fields.items():
            getattr(self.model,name)[:] = values[world]

    def measure(self, qpos, qvel, heading):
        m,d,mj = self.model,self.data,self.mj
        d.qpos[:],d.qvel[:] = qpos,qvel
        mj.mj_fwdPosition(m,d)
        mj.mj_fwdVelocity(m,d)
        mj.mj_subtreeVel(m,d)
        np.testing.assert_array_equal(d.qpos,qpos)
        np.testing.assert_array_equal(d.qvel,qvel)
        assert d.time==0.
        origin = d.site_xpos[self.sites].mean(axis=0)
        com = d.subtree_com[self.root]-origin
        sites = (d.site_xpos[self.sites]-origin)@heading
        floor_z = d.geom_xpos[self.engine.floor,2]
        clear = [float(((vertices@d.geom_xmat[geom].reshape(3,3).T+d.geom_xpos[geom])[:,2].min()-floor_z)*1000)
                 for geom,vertices in zip(self.feet,self.vertices,strict=True)]
        contact = [[],[]]
        for i in range(d.ncon):
            entry = d.contact[i]
            a,b = map(int,entry.geom)
            geom = b if a==self.engine.floor else a if b==self.engine.floor else -1
            if geom in self.feet and entry.efc_address>=0:
                contact[self.feet.index(geom)].append(float((entry.pos-origin)@heading))
        points = contact[0]+contact[1]
        fore = float(com@heading)
        front = (max(points)-fore)*1000 if points else np.nan
        rear = (fore-min(points))*1000 if points else np.nan
        lateral = np.array([-heading[1],heading[0],0.])
        values = [authority.tilt(qpos[3:7]),self.engine.upright_gain(qpos,qvel),
            d.subtree_angmom[self.root]@lateral,1000*(d.subtree_linvel[self.root]@heading),
            1000*fore,1000*sites[0],1000*sites[1],*clear,front,rear]
        return np.asarray(values),np.array([len(c) for c in contact],dtype=np.int16)


def stats(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(count=0,median=None,positive=0)
    return dict(count=len(finite),median=float(np.median(finite)),positive=int((finite>0.).sum()))


def paired_metrics(values, contacts, native, worlds, measured_times, plan):
    by_world = {w:i for i,w in enumerate(worlds)}
    matches = [(by_world[int(a)],by_world[b]) for a,b in plan['archive_pairs'].items()]
    offsets = {**plan['fixed_release_offsets_ms'],
               **{f'load{t:+d}':t for t in plan['bilateral_load_aligned_offsets_ms']}}
    data = {}
    for name,delta in offsets.items():
        def row(i):
            offset = delta + (measured_times[i] if name.startswith('load') else 0)
            k = plan['original_release_control'] + offset//plan['step_ms'] - START
            assert 0<=k<=END-START
            value = values[k,i].copy()
            if not np.array_equal(contacts[k,i]>0,native[k,i]):
                value[-2:] = np.nan
            return value
        diffs = np.stack([row(ia)-row(ib) for ia,ib in matches])
        data[name] = {metric:stats(diffs[:,i]) for i,metric in enumerate(METRICS)}
    return data


def evaluate():
    project = read(PROJECT)
    stage = project['goal_execution'][KEY]
    assert experiment.parent.allowed(project,STAGE) and stage['status']=='archive-ready'
    assert not project['staged_training_request']['simulation_running']
    plan = read(PLAN,stage['protocol_sha256'])
    for path,digest in plan['sources'].items():
        assert sha(ROOT/path)==digest,path
    prior,score,batch,old = (read(ROOT/plan[tag]) for tag in ('earlier_capture_protocol','earlier_capture_result',
        'earlier_capture_collection','geometry_authority_protocol'))
    for path,digest in old['sources'].items():
        assert sha(ROOT/path)==digest,path
    source,meta = (read(ROOT/old[tag]) for tag in ('prior_experiment','load_metadata'))
    geom = Geometry(authority.FixedStateEngine(old,source,meta))
    folder = ROOT/prior['record']
    worlds = plan['worlds']
    prefix = measured.read_trace(folder/'shared',batch['prefix']['chunks'],1,experiment.START,('qpos',),sha)
    raw_side = rotation(prefix['qpos'][-1,worlds,3:7])[...,:2,1]
    length = np.linalg.norm(raw_side,axis=1)
    assert (length>.75).all(),length.min()
    headings = np.stack((raw_side[:,1],-raw_side[:,0],np.zeros(len(worlds))),axis=1)/length[:,None]
    arrays, reports = {},{}
    idx = slice(START-experiment.START-1,END-experiment.START)
    for role in plan['roles']:
        part = batch['branches'][role]
        slice_data = measured.read_trace(folder/role,part['chunks'],experiment.START+1,experiment.END,FIELDS,sha)
        q,v,native = (slice_data[key][idx,worlds] for key in FIELDS)
        row = score['roles'][role]
        assert sha(ROOT/row['source'])==row['source_sha256']
        with np.load(ROOT/row['source'],allow_pickle=False) as raw:
            channels = [prior['forward_support']['selected_worlds'].index(world) for world in worlds]
            np.testing.assert_array_equal(raw['orientation_after'][4::5][idx,channels],q[...,3:7])
        values = np.empty((len(q),len(worlds),len(METRICS)),np.float64)
        contacts = np.empty((len(q),len(worlds),2),np.int16)
        for i,world in enumerate(worlds):
            geom.select(world)
            for k in range(len(q)):
                values[k,i],contacts[k,i] = geom.measure(q[k,i],v[k,i],headings[i])
        agreement = float(((contacts>0)==native).all(axis=-1).mean())
        assert agreement>=plan['contact_parity_minimum'],(role,agreement)
        assert np.isfinite(values[...,:-2]).all()
        times = [row['selected_world_readback'][str(w)]['first_bilateral_support_ms'] for w in worlds]
        assert all(t is not None for t in times)
        values_path = REPORT/f'{role}.npz'
        np.savez_compressed(values_path,values=values,foot_contact_points=contacts,original_foot_flags=native,
            worlds=np.array(worlds),metrics=np.asarray(METRICS),controls=np.arange(START,END+1),
            first_bilateral_support_ms=np.asarray(times),headings_world=headings)
        arrays[role] = dict(path=rel(values_path),sha256=sha(values_path),geometry_contact_parity=agreement)
        reports[role] = paired_metrics(values,contacts,native,worlds,times,plan)
    result = REPORT/'analysis.json'
    write(result,dict(status='verified-archived-state-geometry',protocol_sha256=stage['protocol_sha256'],
        arrays=arrays,paired_missed_minus_admitted=reports,reconstructed_states=len(worlds)*len(plan['roles'])*(END-START+1),
        fixed_heading='Forward is reconstructed from horizontal cross-product of release trunk left-axis and world up.',
        support_margin_scope='Active CPU foot-plane constraint points and sagittal COM gap; GPU normal load used only for timing.',
        new_physical_steps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    stage.update(status='reviewed',result=rel(result),result_sha256=sha(result))
    experiment.prior.training.earlier.selected(project,STAGE,PLAN,STAGE+'-reviewed',result,
        'Select an earlier dynamically testable input using the verified support geometry and upright rate.')
    write(PROJECT,project)
    print('FORWARD_GEOMETRY_VERIFIED',sha(result),{role:r['geometry_contact_parity'] for role,r in arrays.items()},flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','evaluate'))
    option = cli.parse_args().action
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'evaluate':evaluate}[option]()


if __name__ == '__main__':
    main()
