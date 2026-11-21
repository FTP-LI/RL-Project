"""Run the accepted controllers against one continuous scene and one clock."""

from types import SimpleNamespace

import numpy as np
import torch

from roll_stand_walk_sequence import FORWARD as POST_ROLL_READY, Sequence as RollSequence
from roll_stand_walk_sequence import ROLL, STAND_LANDING
from roll_screen_stop_guard import check_state
from routed_getup_sequence import FORWARD as COMMAND, GETUP, STAND, ForwardFeedback
from routed_seven_turn_feedback_controller import SevenCommandTurnFeedbackController
from run_continuous_roll_stand_walk import PolicyControl


FAILED, ROLLING, ROLL_STANDING = -1, 3, 4


class Composition:
    def __init__(self, spec, actors, device):
        self.design = spec['composition']
        self.worlds = self.design['worlds']
        self.device = device
        self.roll_route = RollSequence(spec['roll_route'])
        self.roll_entry = np.zeros(self.worlds, bool)
        self.roll_ready = self.roll_entry.copy()
        self.stand_reads = np.zeros(self.worlds, np.int32)
        plant, walking, route = (spec[key] for key in ('environment','walking','seven_command'))
        engine = spec['sequence']['controller']
        roll_home, official_home = (spec['policy_home'][role] for role in ('official-forward-roll','forward-official'))
        standing_home = spec['switch_policies']['stand']['home']
        home, bounds = (torch.as_tensor(plant[key],device=device,dtype=torch.float32) for key in ('model_home','joint_limits'))
        self.commands = SevenCommandTurnFeedbackController(self.worlds,device,engine,
            [plant['model_home'],standing_home,standing_home],plant['joint_limits'],ForwardFeedback(walking,self.worlds),
            commands=route['commands'],groups=route['groups'],routes=route['routes'],ramp=spec['ramped_forward']['ramp'],
            release=self.design['release_control'],stand_at=self.design['getup_to_stand_control'],
            forward_at=self.design['command_entry_control'])
        self.command_actors = {GETUP:actors['getup'],STAND:actors['stand-and-base'],COMMAND:{
            'base':actors['stand-and-base'],'official':actors['forward-official'],'ppo-candidate':actors['turn-right']}}
        assert set(self.commands.route_masks)==set(self.command_actors[COMMAND])
        self.roll = PolicyControl(spec['roll_route'],{'roll':actors['official-forward-roll'],'walk':actors['forward-official']},
            {role:torch.as_tensor(value,device=device,dtype=torch.float32) for role,value in (('roll',roll_home),('walk',official_home))},
            home.expand(self.worlds,-1),bounds)

    def observe_roll(self, control, frame):
        d = self.design
        if not d['getup_to_stand_control'] <= control <= d['roll_to_base_stand_control']:
            return
        # The first standing checkpoint is the original first roll-route state.
        # The get-up controller owns this prefix until the original roll entry.
        if control == d['roll_start_control']:
            self.roll_entry = self.commands.mode.detach().cpu().numpy() == STAND
        state = dict(frame)
        if control >= d['roll_start_control']:
            state['active'] = frame['active'] & self.roll_entry
            safety = check_state(state,control,self.roll_route.plan)
            assert not safety['stop'], safety
        self.roll_route.update(control-d['getup_to_stand_control'],state)
        if control == d['roll_to_base_stand_control']:
            self.roll_ready = (self.roll_route.forward_at >= 0) & (self.roll_route.phase == POST_ROLL_READY)

    def observe_base_stand(self, control, frame):
        d = self.design
        if d['roll_to_base_stand_control'] < control <= d['command_entry_control']:
            stable = self.roll_route.stable(frame) & self.roll_ready
            self.stand_reads = np.where(stable,self.stand_reads+1,0)

    def action(self, control, sensed, previous):
        d, bank = self.design, self.commands
        start, end, command = (d[key] for key in ('roll_start_control','roll_to_base_stand_control','command_entry_control'))
        if start <= control < end:
            if control == start:
                self.roll.previous.copy_(previous)
            phase = self.roll_route.phase
            # The original landing dwell is certified per world. Hold the
            # same official stance and raw history until the common next entry.
            roll_stage = SimpleNamespace(phase=np.where(phase==POST_ROLL_READY,STAND_LANDING,phase),
                                         phase_changed=np.where(phase==POST_ROLL_READY,-1,self.roll_route.phase_changed+d['getup_to_stand_control']))
            zero = sensed.clone()
            zero[:,48:] = 0.
            target,row = self.roll.next(control+1,zero,roll_stage)
            mode = np.where(roll_stage.phase==ROLL,ROLLING,np.where(roll_stage.phase==STAND_LANDING,ROLL_STANDING,FAILED))
            return SimpleNamespace(raw=torch.as_tensor(row['raw_policy_actions'],device=self.device), target=target,
                requested=torch.as_tensor(row['requested_targets'],device=self.device),mode=mode, bridge=row['bridge'],
                actor_input=row['actor_input'], alternate=row['alternate_action'], yaw=row['feedback_yaw'])

        if control == end:
            bank.mode.fill_(FAILED)
            bank.mode[torch.as_tensor(self.roll_ready,device=self.device)] = STAND
            bank.dwell.zero_()
            # This standing policy is different from the official roll exit.
            # Clear its old pre-roll history before the bounded base handoff.
            bank.history[STAND].zero_()
            bank.history[COMMAND].zero_()
        if control == command:
            ready = torch.as_tensor(self.stand_reads>=bank.dwell_steps,device=self.device)
            bank.mode[~ready] = FAILED
        x, bridge = bank.advance(control,sensed)
        if end <= control < command:
            bridge = bank.mode==STAND
        raw,yaw,alternate = bank.infer(control,x,self.command_actors)
        return SimpleNamespace(raw=raw,target=None,mode=bank.mode.detach().cpu().numpy().copy(),
                               bridge=bridge,actor_input=x,alternate=alternate,yaw=yaw)

    def apply(self, term, previous, control, action):
        if action.target is not None:
            term.requested_target.copy_(action.requested)
            term.projected_destination.copy_(action.target)
            term._processed_actions.copy_(action.target)
            term.previous_target.copy_(action.target)
        elif control >= self.design['getup_to_stand_control']:
            self.commands.apply(term,previous,action.raw,action.alternate,action.bridge,control,.02)
            failed = self.commands.mode == FAILED
            term._processed_actions[failed] = previous[failed]
            term.previous_target[failed] = previous[failed]
