"""Prepare a separate, reachable head-command curriculum without locking joints.

Why: the archived neck command reaches about 83 degrees while the compiled
neck upper limit is 60 degrees. Preserve zero-centered pose commands and the
existing early curriculum, but cap symmetric amplitudes at the available soft
joint range around HOME. Use the robot configuration's existing soft-limit
factor, rather than inventing a new reward weight or tightening head tracking.
This returns a copied configuration only. It neither runs training nor changes
the archived source, current worker, actuator targets, physics or rewards.
"""
from copy import deepcopy
import math
import re

HEAD_NAMES=('neck_pitch','head_pitch','head_yaw','head_roll')


def prepare_reachable_head_commands(cfg):
    import mujoco
    robot=cfg.scene.entities['robot']
    model=robot.spec_fn().compile()
    factor=float(robot.articulation.soft_joint_pos_limit_factor)
    assert 0 < factor <= 1
    limits=[]
    for name in HEAD_NAMES:
        joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        assert joint >= 0 and model.jnt_limited[joint]
        home_values=[float(value) for pattern,value in robot.init_state.joint_pos.items()
                     if re.fullmatch(pattern,name)]
        assert len(home_values)==1,(name,home_values)
        home=home_values[0];lower,upper=map(float,model.jnt_range[joint])
        midpoint=(lower+upper)/2;radius=(upper-lower)*factor/2
        soft_lower,soft_upper=midpoint-radius,midpoint+radius
        cap=min(home-soft_lower,soft_upper-home)
        assert math.isfinite(cap) and cap>0,(name,home,soft_lower,soft_upper)
        limits.append(dict(joint=name,home_rad=home,hard_range_rad=[lower,upper],
            soft_range_rad=[soft_lower,soft_upper],symmetric_command_cap_rad=cap))
    patched=deepcopy(cfg);changes=[]

    def bounded(ranges,location):
        assert len(ranges)==4
        result=[]
        for interval,bound in zip(ranges,limits,strict=True):
            lo,hi=map(float,interval)
            assert lo<=0<=hi and abs(lo+hi)<1e-8,'Preserve the original zero-centered command distribution'
            amplitude=min(hi,bound['symmetric_command_cap_rad'])
            new=(-amplitude,amplitude);result.append(new)
            if new!=(lo,hi):changes.append(dict(location=location,joint=bound['joint'],before_rad=[lo,hi],after_rad=list(new)))
        return tuple(result)

    command=patched.commands['head_pose']
    command.ranges=bounded(command.ranges,'commands.head_pose.ranges')
    curriculum=patched.curriculum['head_pose_range']
    assert curriculum.params['command_name']=='head_pose'
    for stage in curriculum.params['range_stages']:
        stage['ranges']=bounded(stage['ranges'],f'curriculum.head_pose_range.step_{stage["step"]}')
    return patched,dict(soft_joint_pos_limit_factor=factor,joint_bounds=limits,changes=changes,
        command_mean_preserved_zero=True,head_locked=False,rewards_changed=False,
        actuator_targets_clipped=False,physics_changed=False,training_started=False)
