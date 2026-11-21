"""Record zero-observer refusals without changing the guard decision.

The historical coordinator combines observer and project failure in one
message. This process-local wrapper logs the original decision and a later
read-only project snapshot only on refusal. It never converts refusal to allow.
"""
from contextlib import contextmanager
from datetime import datetime,timezone
import json
from pathlib import Path
from unittest.mock import patch

import native_zero_coordinator as coordinator

ROOT=Path(__file__).resolve().parents[2]


def _record(path,event):
    event['checked_at']=datetime.now(timezone.utc).isoformat()
    try:
        with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
    except OSError as exc:
        print('GUARD_REJECTION_LOG_ERROR',type(exc).__name__,str(exc),flush=True)


def _project_snapshot(plan,wrapper_sha,running):
    """Read after rejection: this cannot reconstruct a transient past state."""
    try:
        project=coordinator.old.read(ROOT/'project.json')
        task=project.get('staged_training_request',{});goal=project.get('goal_execution',{})
        execution=goal.get('native_file_observer_evaluation',{})
        upstream=coordinator.provenance.PLAN;wrapper=coordinator.old.inside(coordinator.WRAPPER)
        return {'goal_status':goal.get('status'),'goal_pause':goal.get('pause_requested_by_user'),
                'task_pause':task.get('pause_requested'),'learning_running':task.get('learning_running'),
                'simulation_running':task.get('simulation_running'),'expected_running':running,
                'stage_matches':task.get('current_stage')==plan.get('stage'),
                'protocol_frozen':task.get('current_protocol_frozen'),
                'protocol_matches':task.get('current_protocol')==upstream.relative_to(ROOT).as_posix(),
                'upstream_digest_matches':coordinator.old.sha(upstream)==coordinator.provenance.PLAN_SHA,
                'wrapper_path_matches':execution.get('native_zero_plan')==coordinator.WRAPPER,
                'wrapper_registered_digest_matches':execution.get('native_zero_plan_sha256')==wrapper_sha,
                'wrapper_digest_matches':coordinator.old.sha(wrapper)==wrapper_sha,
                'scope':'Snapshot taken after original refusal; transient errors may no longer be present.'}
    except (OSError,KeyError,TypeError,ValueError,json.JSONDecodeError) as exc:
        return {'snapshot_error':type(exc).__name__+': '+str(exc)}


@contextmanager
def log_refusals(event_path):
    original_child=coordinator.check_child;original_project=coordinator.project_allows

    def checked_child(child,health_file,**kwargs):
        try:decision=original_child(child,health_file,**kwargs)
        except Exception as exc:
            _record(event_path,{'component':'observer','exception':type(exc).__name__+': '+str(exc)})
            raise
        if not decision.get('allowed'):
            try:health=coordinator.old.read(Path(health_file))
            except (OSError,ValueError,json.JSONDecodeError) as exc:health={'read_error':type(exc).__name__+': '+str(exc)}
            _record(event_path,{'component':'observer','original_decision':decision,'health_after_refusal':health})
        return decision

    def checked_project(plan,wrapper_sha,*,running):
        try:allowed=original_project(plan,wrapper_sha,running=running)
        except Exception as exc:
            _record(event_path,{'component':'project','exception':type(exc).__name__+': '+str(exc)})
            raise
        if not allowed:_record(event_path,{'component':'project','original_allowed':False,
                                           'project_after_refusal':_project_snapshot(plan,wrapper_sha,running)})
        return allowed

    with patch.object(coordinator,'check_child',checked_child),patch.object(coordinator,'project_allows',checked_project):
        yield
