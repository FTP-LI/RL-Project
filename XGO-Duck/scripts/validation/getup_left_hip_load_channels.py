"""Read the left hip and selected contact pairs at the solved physics instant."""

from time import perf_counter

import numpy as np
import torch

from getup_shared_release import cpu


LABELS = ("ground-left-foot","ground-right-foot","head-left-foot","head-right-foot","head-left-hip","head-right-hip")
JOINT = "left_hip_pitch"


def pair_layout(model):
    import mujoco

    floor, = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    head = model.geom("robot/head_collision").id
    lf,rf,lh,rh = [model.geom("robot/"+name+"_collision").id for name in
                   ("left_foot","right_foot","left_hip_yaw","right_hip_yaw")]
    pairs = [(floor,lf),(floor,rf),(head,lf),(head,rf),(head,lh),(head,rh)]
    lut = np.full((model.ngeom,model.ngeom),-1,np.int32)
    for i,(a,b) in enumerate(pairs):
        lut[a,b] = lut[b,a] = i
    joint = model.joint("robot/"+JOINT)
    ids = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                          & (model.actuator_trnid[:,0] == joint.id))
    motor, = ids
    assert model.actuator_gear[motor,0] == 1 and not model.actuator_gear[motor,1:].any()
    assert model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_HINGE
    return dict(pairs=[list(map(int,pair)) for pair in pairs],motor=int(motor),qpos=int(model.jnt_qposadr[joint.id]),
                dof=int(model.jnt_dofadr[joint.id]),cone=int(model.opt.cone),geom_count=int(model.ngeom)),lut


def reduce_contacts(world,channel,normal,n):
    assert np.isfinite(normal).all() and (normal >= -1e-6).all()
    assert ((world >= 0)&(world < n)).all() and ((channel >= 0)&(channel < len(LABELS))).all()
    count = np.zeros((n,len(LABELS)),np.int16)
    force = np.zeros((n,len(LABELS)),np.float64)
    np.add.at(count,(world,channel),1)
    np.add.at(force,(world,channel),normal)
    return count,force.astype(np.float32)


class SolvedLoad:
    def __init__(self,env,names,window):
        self.env = env
        self.layout,lut = pair_layout(env.sim.mj_model)
        self.channels = torch.from_numpy(lut).to(env.device)
        self.column = names.index(JOINT)
        self.term = env.action_manager.get_term("joint_pos")
        self.first,self.last = (round(t/env.physics_dt) for t in window)
        self.rows,self.raw = [],[]
        self.read_seconds = 0.
        self.capacity = int(env.sim.wp_data.contact.dist.shape[0])
        self.peak = 0

    def contacts(self):
        import mujoco_warp as warp_mujoco
        import warp as wp

        sim = self.env.sim
        data = sim.wp_data
        m = int(data.nacon.numpy()[0])
        assert 0 <= m < self.capacity,"Native contact capacity exhausted"
        self.peak = max(self.peak,m)
        pairs = wp.to_torch(data.contact.geom)[:m]
        assert not ((pairs < 0)|(pairs >= self.layout["geom_count"])).any()
        kind = self.channels[pairs[:,0].long(),pairs[:,1].long()]
        dist = wp.to_torch(data.contact.dist)[:m]
        physical = (wp.to_torch(data.contact.type)[:m]&1) != 0
        ids = torch.nonzero((kind >= 0)&physical&(dist < 0),as_tuple=True)[0]
        world = wp.to_torch(data.contact.worldid)[ids].long()
        addresses = wp.to_torch(data.contact.efc_address)[ids].long()
        dimensions = wp.to_torch(data.contact.dim)[ids]
        nforce = int(data.njmax)
        raw_efc = wp.to_torch(data.efc.force)
        cols = torch.arange(10,device=ids.device)
        if ids.numel():
            # Read only selected physical pairs so the 3,072-world capture
            # does not transfer the entire Warp contact arena each substep.
            stream = wp.stream_from_torch(torch.cuda.current_stream(self.env.device))
            selected = ids.to(torch.int32).contiguous()
            with wp.ScopedStream(stream):
                forces = wp.empty(len(ids),dtype=wp.spatial_vector,device=sim.wp_device)
                warp_mujoco.contact_force(sim.wp_model,data,wp.from_torch(selected),False,forces)
            normal = wp.to_torch(forces)[:,0]
        else:
            normal = torch.zeros(0,device=ids.device)
        base = addresses[:,0]
        efc_index = base[:,None]+cols
        valid = (base[:,None] >= 0)&(efc_index < nforce)
        sample = torch.where(valid,raw_efc[world[:,None],efc_index.clamp(0,nforce-1)],0.)
        values = dict(world=cpu(world),channel=cpu(kind[ids]),pairs=cpu(pairs[ids]),distance=cpu(dist[ids]),
            dim=cpu(dimensions),address=cpu(addresses),constraint_rows=cpu(sample),normal=cpu(normal))
        return reduce_contacts(values["world"],values["channel"],values["normal"],self.env.num_envs),values

    def advance(self,original):
        env = self.env
        k = env._sim_step_counter
        if not self.first < k <= self.last:
            return original()
        data = env.sim.data
        q,dof,motor = (self.layout[key] for key in ("qpos","dof","motor"))
        start = perf_counter()
        target = cpu(torch.stack((self.term.requested_target[:,self.column],self.term._processed_actions[:,self.column],
            data.qpos[:,q],data.qvel[:,dof],data.ctrl[:,motor]),dim=-1))
        q_before = cpu(data.qpos[:,3:7])
        pre = perf_counter()-start
        original()
        start = perf_counter()
        (points,loads),contacts = self.contacts()
        response = cpu(torch.stack((data.qpos[:,q],data.qvel[:,dof],data.actuator_force[:,motor],
            data.qfrc_actuator[:,dof],data.qfrc_constraint[:,dof]),dim=-1))
        self.rows.append(dict(substep=k,target=target,response=response,quaternion_before=q_before,points=points,load_n=loads))
        self.raw.append(contacts)
        self.read_seconds += pre+perf_counter()-start

    def save(self,record):
        assert len(self.rows) == len(self.raw) == self.last-self.first
        archive = record/"load-clock.npz"
        np.savez_compressed(archive,**{key:np.stack([row[key] for row in self.rows]) for key in self.rows[0]})
        edges = np.r_[0,np.cumsum([len(row["world"]) for row in self.raw])]
        raw = record/"solved-contact-pairs.npz"
        np.savez_compressed(raw,edges=edges,**{key:np.concatenate([row[key] for row in self.raw]) for key in self.raw[0]})
        return dict(archive=archive.name,raw=raw.name,labels=list(LABELS),capacity=self.capacity,captured_peak=self.peak,
                    layout=self.layout,first_substep=self.first+1,last_substep=self.last,read_seconds=self.read_seconds)
