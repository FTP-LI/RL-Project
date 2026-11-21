"""Build native fixtures from verified first-episode policy inputs."""

import numpy as np
import torch

from getup_second_attempt import SecondAttempt
from s1_reference_inputs import ROOT, read, sha


def controller(n,device,screen,settings,env,neck):
    return SecondAttempt(n,device,screen,settings,env["joint_limits"],env["simulation_home"],neck,
                         env["criteria"]["release_alignment_min"])


def ready_fixture(screen,settings,env,neck):
    state = controller(4,"cpu",screen,settings,env,neck)
    initial = torch.zeros(4,61)
    initial[:,3] = -1.
    blocked = initial.clone()
    blocked[:,3] = -.5
    blocked[:,5] = -.8660254
    angle = torch.zeros(4)
    for k in range(state.release,state.first+1):
        state.observe(k,initial if k == state.release else blocked,angle)
    assert torch.equal(state.trigger,torch.full_like(state.trigger,state.first))
    saved = state.snapshot()
    quiet = initial.clone()
    quiet[2,0] = 2.
    quiet[3,3:6] = blocked[3,3:6]
    angle[:] = torch.tensor([.10,.04,.10,.10])
    end = state.first+state.duration
    for k in range(state.first+1,end+1):
        state.observe(k,quiet,angle)
    assert state.ready.tolist() == [end,-1,-1,-1]
    before = torch.broadcast_to(state.retreat,(4,14)).clone()
    request = state.limits[:,1].expand_as(before).clone()
    actual = state.targets(end,before,request,request)
    torch.testing.assert_close(actual,before+(request-before).clamp(0.,state.rate*state.dt))
    state.restore(saved)
    assert state.boundary == state.first and state.ready.eq(-1).all()
    for value,original in zip(state.snapshot()[1],saved[1],strict=True):
        torch.testing.assert_close(value,original,atol=0.,rtol=0.,equal_nan=True)
    return dict(ready_at_original_boundary=end,no_ready_without_neck_withdrawal=True,
        rejects_unquiet_or_non_supine=True,return_to_policy_rate_limited=True,controller_snapshot_exact=True)


def rehearsal(proof,screen,settings,neck):
    fields = ("qpos","qvel","joint_positions","actor_input","observations","raw_policy_actions",
              "requested_targets","applied_targets","time")
    fixtures = {key:[] for key in ("first","last")}
    results = []
    duration = settings["controls"]
    for row in proof["results"]:
        record = ROOT/row["record"]
        p = read(record/"protocol.json")["environment"]
        trace = record/f'trace-{row["role"]}.npz'
        assert sha(trace) == row["trace_sha256"]
        n = row["worlds"]
        state = controller(n,"cpu",screen,settings,p,neck)
        stop = state.last+duration+1
        length = stop-state.release+1
        with np.load(trace,allow_pickle=False) as src:
            t = {key:src[key][:length].copy() for key in fields}
        np.testing.assert_allclose(t["time"],np.arange(state.release+1,stop+2)*state.dt,atol=1e-8,rtol=0.)
        np.testing.assert_array_equal(t["actor_input"][...,:6],t["observations"][...,:6])
        reference_trigger = np.full(n,-1)
        for item in row["triggers"]:
            reference_trigger[item["world"]] = item["first_decision_control"]
        previous = t["applied_targets"][0].copy()
        dest,limits = state.retreat.numpy(),state.limits.numpy()
        largest = 0.
        for i,k in enumerate(range(state.release,stop+1)):
            source = torch.from_numpy(t["actor_input"][i])
            measured = torch.from_numpy(t["joint_positions"][max(i-1,0),:,neck])
            state.observe(k,source,measured)
            if k < state.first:
                previous = t["applied_targets"][i].copy()
                continue
            desired,ordinary = t["requested_targets"][i],t["applied_targets"][i]
            actual = state.targets(k,*(torch.from_numpy(x) for x in (previous,desired,ordinary))).numpy()
            expected_active = (reference_trigger >= 0)&(reference_trigger <= k)
            expected_retreat = expected_active&(k-reference_trigger < duration)
            delta = np.clip(np.clip(np.where(expected_retreat[:,None],dest,desired),*limits.T)-previous,
                            -state.dt*state.rate,state.dt*state.rate)
            np.testing.assert_array_equal(actual,np.where(expected_active[:,None],previous+delta,ordinary))
            largest = max(largest,float(abs((actual-previous)[expected_active]).max(initial=0.)))
            for name,offset in (("first",0),("last",duration-1)):
                for w in np.flatnonzero(expected_active&(k-reference_trigger == offset)):
                    fixtures[name].append(dict(seed=row["seed"],world=w,boundary=k,trigger=reference_trigger[w],
                        qpos=t["qpos"][i-1,w],qvel=t["qvel"][i-1,w],measured=t["joint_positions"][i-1,w],
                        previous=previous[w],raw=t["raw_policy_actions"][i,w],requested=desired[w],candidate=actual[w]))
            previous = actual
        np.testing.assert_array_equal(state.trigger.numpy(),reference_trigger)
        assert largest <= .040001
        results.append(dict(seed=row["seed"],worlds=n,triggers=int((reference_trigger >= 0).sum()),max_target_step_rad=largest))
    count = sum(row["triggers"] for row in results)
    assert count == len(fixtures["first"]) == len(fixtures["last"]) == 24
    arrays = {f"{role}__{key}":np.stack([entry[key] for entry in items])
              for role,items in fixtures.items() for key in items[0]}
    return results,arrays,ready_fixture(screen,settings,p,neck)
