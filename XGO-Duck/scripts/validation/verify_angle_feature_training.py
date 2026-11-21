"""Add real nonzero-angle parity and column checks to the frozen verifier."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from s1_reference_inputs import ROOT,read,sha
from record_io import write
import verify_roll_top_contact_training as base

OUT=ROOT/'records/reports/roll-angle-feature'


def verify(planfile):
    base.OUT=OUT
    result=base.verify(planfile)
    plan=read(Path(planfile));record=ROOT/'records/experiments'/plan['run_id']
    contract=read(record/'angle-input-contract.json')
    assert contract['status']=='passed' and contract['physical_steps']==0
    assert contract['expanded_initial_sha256']==sha(record/'training-expanded-initial.npz')
    assert contract['drift_probe_sha256']==sha(record/'phase-drift-probe.npz')
    trigger=read(record/'learned-clip-trigger.json')
    assert trigger['common_step']==700 and trigger['actor_updates_completed']>=15 and trigger['extra_physics_steps']==0
    clips=read(record/'clips.json')
    assert any(c['complete_clip'] and c['common_steps'][0]==700 and c['common_steps'][-1]==850 for c in clips)
    state=torch.load(result['checkpoint'],map_location='cpu',weights_only=True)['actor_state_dict']
    column=state['mlp.0.weight'][:,55]
    if plan['arm']=='control':assert not torch.count_nonzero(column).item()
    else:assert torch.count_nonzero(column).item()>0, 'Candidate did not learn its new input'
    assert state['obs_normalizer._mean'][0,55]==0 and state['obs_normalizer._std'][0,55]==1
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    actor.load_state_dict(state);actor.eval();actor.obs_normalizer.until=0
    with np.load(record/'phase-drift-probe.npz') as f:obs=f['observations'].copy()
    assert np.count_nonzero(abs(obs[:,55])>1)>32
    with torch.no_grad():actual=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])).numpy()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/result['model']),sess_options=options,providers=['CPUExecutionProvider'])
    metadata=json.loads(session.get_modelmeta().custom_metadata_map['roll_angle_contract'])
    assert metadata['slot']==55 and metadata['actor_angle_access']==(plan['arm']=='candidate')
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    assert not torch.cuda.is_initialized()
    result.update(angle_input_contract=contract,learned_policy_clip_verified=True,
        angle_column_nonzero=int(torch.count_nonzero(column)),nonzero_phase_parity_inputs=len(obs),
        maximum_angle_export_error=float(abs(actual-expected).max()),angle_verifier_sha256=sha(Path(__file__)))
    write(OUT/plan['arm']/'training-verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='angle_input_contract'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('protocol');verify(p.parse_args().protocol)
