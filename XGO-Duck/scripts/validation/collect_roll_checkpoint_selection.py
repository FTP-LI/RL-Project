"""Swap only the roll actor at the restored standing boundary."""

from unittest.mock import patch

import numpy as np

import collect_roll_neck_support_pair as paired_roll
from roll_entry_snapshot import RollEntrySnapshot
from run_roll_stand_forward_diagnostic import ROOT, require, sha


CANDIDATE = 'earlier-checkpoint-candidate'
REFERENCES = ('first-final-checkpoint-reference','second-final-checkpoint-reference')
ROLES = (*REFERENCES,CANDIDATE)
LABELS = {
    'shared':'共同站立', REFERENCES[0]:'第50次保存点第一次参照', REFERENCES[1]:'第50次保存点第二次参照',
    CANDIDATE:'第30次保存点候选',
}


def make_collector(base, may_continue):
    sample = paired_roll.make_collector(base, may_continue)

    def collect(env, obs, plan, folder, endpoint, networks, homes, sim_home, limits):
        design = plan['checkpoint_selection']
        shared = design['plant']['shared_snapshot_control']
        require(env.common_step_counter == env._sim_step_counter == 0 and may_continue(), 'Actor admission is not an authorized CUDA zero step')
        nominated = plan['earlier_roll']
        final = networks['roll']
        early = base.make_actor(nominated,env.device)
        session = base.onnx_session(ROOT/nominated['model'])
        final_session = base.onnx_session(ROOT/plan['roll']['model'])
        metadata = session.get_modelmeta().custom_metadata_map
        require(metadata==final_session.get_modelmeta().custom_metadata_map,'Checkpoint joint order, defaults or interface changed')
        observations = base.model_input(obs['actor'],sim_home,homes['roll'],base.torch.zeros_like(sim_home))
        expected = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in base.cpu(observations)])
        predicted = base.cpu(early(base.td(observations)))
        np.testing.assert_allclose(predicted,expected,atol=2e-5,rtol=2e-5)
        base.write(folder/'checkpoint-actor-admission.json',dict(physical_controls=0,physical_substeps=0,actual_cuda_inputs=len(expected),
            earlier_on_actual_zero_max_abs_error_rad=float(np.abs(predicted-expected).max()),
            joint_names=metadata['joint_names'].split(','),models={name:meta['model_sha256']
                for name,meta in (('final',plan['roll']),('earlier',nominated))}))

        activated = []
        actor_for = {**dict.fromkeys(REFERENCES,final),CANDIDATE:early}
        model_for = {**dict.fromkeys(REFERENCES,plan['roll']),CANDIDATE:nominated}

        class CheckpointSnapshot(RollEntrySnapshot):
            def __init__(self, raw, policy, budget):
                self.policy = policy
                self.standing_fork = raw.common_step_counter==shared
                super().__init__(raw,policy,budget)

            def restore(self):
                restored = super().restore()
                if self.standing_fork:
                    require(len(activated)<len(ROLES) and restored['counters']==[shared,5*shared], 'Unexpected actor activation boundary')
                    role = ROLES[len(activated)]
                    require(sha(ROOT/model_for[role]['model'])==model_for[role]['model_sha256'], 'Roll actor changed since admission')
                    self.policy.networks['roll'] = actor_for[role]
                    activated.append(role)
                    base.write(folder/'clips'/role/'roll-actor.json',dict(role=role,model_sha256=model_for[role]['model_sha256'],
                        restore_control=shared,first_roll_actor_control=design['plant']['first_roll_actor_control']))
                return restored

        # Use the already verified paired recorder, contact guards and first
        # substep checks. All physical target offsets are zero in this test.
        runner_plan = dict(plan,neck_support=dict(plant=design['plant'],roles=dict.fromkeys(ROLES,0.),
            offset={'joint':'neck_pitch'},assessment=design['assessment']))
        with patch.multiple(paired_roll,RollEntrySnapshot=CheckpointSnapshot,LABELS=LABELS):
            sample(env,obs,runner_plan,folder,endpoint,networks,homes,sim_home,limits)
        require(tuple(activated)==ROLES,'Checkpoint activation order differs')
    return collect
