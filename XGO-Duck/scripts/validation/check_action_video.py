"""CPU recorder lifecycle checks, with no images, encoder or simulation."""
from pathlib import Path
from types import SimpleNamespace
import sys
import json
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from action_video import ActionVideo,recording_plan,POLICY
from training_watch import SparseVideo
from s1_reference_inputs import sha
from record_io import write


def main():
    assert not torch.cuda.is_initialized()
    assert recording_plan('getup',8.,.02)['video_seconds']==8.
    assert recording_plan('getup',11.,.02)['video_seconds']==11.
    assert recording_plan('roll',7.,.02)['video_seconds']==7.
    assert recording_plan('roll',9.,.02)['video_seconds']==9.
    assert recording_plan('walk',20.,.02)['video_seconds']==4.
    assert recording_plan('switch',60.,.02,23.)['video_seconds']==23.
    events=[];env=SimpleNamespace(common_step_counter=0,episode_length_buf=torch.tensor([0]),step_dt=.02,
        cfg=SimpleNamespace(episode_length_s=8.),recorder_manager=SimpleNamespace(record_pre_reset=lambda ids:events.append('reset')))
    # Avoid creating an encoder: exercise the real boundary and reset-hook
    # methods with a fake base sample that only records call order and times.
    original_init,original_sample,original_finish=SparseVideo.__init__,SparseVideo.sample,ActionVideo.finish
    def initialize(self,snapshot,config,record):
        self.env=snapshot.env;self.cfg=config;self.writer=None;self.last_started=None
        self.horizon=400;self.stride=2;self.frame_steps=[];self.clips=[]
    def sample(self):
        if self.writer is None:self.writer=object();self.start_step=self.env.common_step_counter
        self.frame_steps.append(self.env.common_step_counter);events.append(('frame',self.env.common_step_counter))
    def finish(self,complete=False):
        events.append(('finish',complete));self.writer=None
    SparseVideo.__init__,SparseVideo.sample,ActionVideo.finish=initialize,sample,finish
    try:
        video=ActionVideo(SimpleNamespace(env=env),'getup',Path('.'))
        env.common_step_counter=250;env.episode_length_buf.fill_(250);video.sample()
        assert events==[],'Due recording must wait for the start of an action'
        env.common_step_counter=400;env.episode_length_buf.zero_();video.sample()
        assert events==[('frame',400)]
        env.common_step_counter=800;env.episode_length_buf.fill_(400)
        env.recorder_manager.record_pre_reset(torch.tensor([0]))
        assert events==[('frame',400),('frame',800),('finish',True),'reset']
        video.close();assert env.recorder_manager.record_pre_reset is not None
    finally:SparseVideo.__init__,SparseVideo.sample,ActionVideo.finish=original_init,original_sample,original_finish
    assert not torch.cuda.is_initialized()
    out=ROOT/'records/reports/action-video';out.mkdir(exist_ok=True)
    result=dict(status='passed',actual_train_and_eval_horizons_checked=True,mid_action_start_rejected=True,
        terminal_frame_before_reset=True,full_sequence_duration_checked=True,simulated_callback_order=events,
        no_images_generated=True,physics_steps=0,optimizer_steps=0,cuda_initialized=False,
        future_learner_must_use_ActionVideo=True,policy_sha256=sha(POLICY),source_sha256=sha(ROOT/'scripts/training/action_video.py'),script_sha256=sha(Path(__file__)))
    write(out/'preflight.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
