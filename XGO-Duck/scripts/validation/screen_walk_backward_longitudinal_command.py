"""Screen a stronger internal backward command without integrating physics."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from screen_walk_backward_lateral_command import CAP, HISTORY, OUT, PLAN, cosine, donor
from s1_reference_inputs import ROOT, read, sha
from record_io import write


LEVELS = (-.16,-.18,-.20,-.22,-.24,-.28)
DEST = OUT/'longitudinal-command-target-screen.json'


def main():
    assert not DEST.exists()
    prior = OUT/'lateral-command-target-screen.json'
    old,history,protocol = read(prior),read(HISTORY),read(PLAN)
    assert old['status']=='closed-static-target-screen' and not old['heldout_used'] and not protocol['heldout_used']
    assert old['protocol_sha256']==sha(PLAN) and old['history_sha256']==sha(HISTORY)
    plant = protocol['environment']
    axes = history['axes']
    indices = [plant['joint_names'].index(f'{side}_{axis}') for side in ('left','right') for axis in axes]
    directions = {str(row['seed']):np.median(donor(row['roles']['backward']),axis=0) for row in history['cohorts']}
    for key,value in directions.items():
        np.testing.assert_array_equal(value,old['directions'][key])
    model = ROOT/plant['model']
    assert sha(model)==plant['model_sha256']
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider'])
    sources = {path.relative_to(ROOT).as_posix():sha(path) for path in (prior,HISTORY,PLAN,model,Path(__file__))}

    def infer(inputs):
        return np.concatenate([session.run(None,{'obs':obs[None]})[0] for obs in inputs])

    home = np.asarray(plant['model_home'],np.float32)
    low,high = np.asarray(plant['joint_limits'],np.float32).T
    results = {}
    for seed in protocol['runs']:
        proof_path = OUT/f'seed{seed}-analysis.json'
        proof = read(proof_path)
        assert proof['protocol_sha256']==sha(PLAN) and not proof['heldout_used']
        archive = ROOT/proof['record']/'trace-backward.npz'
        assert sha(archive)==proof['trace_sha256']['backward']
        for path in (proof_path,archive):
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
        worlds = proof['common_near_static']
        with np.load(archive) as data:
            observed = data['observations'][0,worlds].copy()
            saved = data['actions'][0,worlds]
        np.testing.assert_allclose(observed[:,48:51],np.broadcast_to([-.14,0.,0.],(len(worlds),3)),atol=2e-8,rtol=0)
        nominal = infer(observed)
        np.testing.assert_allclose(nominal,saved,atol=2e-5,rtol=2e-5)
        base = np.clip(nominal+home,low,high)
        values = {}
        for level in LEVELS:
            actor_input = observed.copy()
            actor_input[:,48] = level
            candidate = np.clip(infer(actor_input)+home,low,high)
            delta = np.clip(candidate-base,-CAP,CAP)[:,indices]
            magnitude = np.linalg.norm(delta,axis=-1)
            align = {key:cosine(delta,vec) for key,vec in directions.items()}
            passed = (np.median(magnitude)>=old['required_minimum_leg_delta_norm_rad'] and
                      all(np.median(a)>=old['required_median_cosine'] and
                          (a>0).mean()>=old['required_positive_cosine_fraction'] for a in align.values()))
            values[str(level)] = dict(passed=bool(passed),median_leg_target_delta_rad=float(np.median(magnitude)),
                median_cosine={key:float(np.median(value)) for key,value in align.items()},
                positive_fraction={key:float((value>0).mean()) for key,value in align.items()})
        results[str(seed)] = dict(worlds=worlds,levels=values)
    admitted = [level for level in LEVELS if all(row['levels'][str(level)]['passed'] for row in results.values())]
    write(DEST,dict(status='passed-static-target-screen' if admitted else 'closed-static-target-screen',
        source_sha256=sources,target_cap_rad=CAP,baseline_internal_m_s=-.14,admitted_levels=admitted,
        source_gates={key:value for key,value in old.items() if key.startswith('required_')},batches=results,
        learning_updates=0,physical_substeps=0,heldout_used=False,
        scope='Change internal forward-axis input only on saved backward stalls. External backward command and original action gates remain unchanged.'))
    print('BACKWARD_LONGITUDINAL_COMMAND_SCREEN',admitted,
          {seed:{level:(row['median_leg_target_delta_rad'],min(row['median_cosine'].values()))
                 for level,row in batch['levels'].items()} for seed,batch in results.items()},flush=True)


if __name__=='__main__':
    main()
