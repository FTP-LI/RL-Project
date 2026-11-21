"""Reuse canonical file paths while rereading every permission and digest.

WSL traversal of /mnt/c is expensive at every PPO minibatch. Resolve the fixed
paths once for an admitted run; do not cache project values or file contents.
Only the canonical project-relative protocol spelling is accepted.
"""
from pathlib import Path
import hashlib
import json


class CanonicalAuthorization:
    def __init__(self, root, project, protocol, protocol_sha256, stage, run_id,
                 command, supervisor_source, parent_guard, read, sha):
        self.root = Path(root).resolve()
        self.project = Path(project).resolve()
        self.protocol = Path(protocol).resolve()
        if not self.protocol.is_relative_to(self.root):
            raise ValueError("Protocol outside project")
        self.protocol_name = self.protocol.relative_to(self.root).as_posix()
        self.digest, self.stage, self.run_id = protocol_sha256, stage, run_id
        self.command = list(command)
        self.supervisor_source = Path(supervisor_source).resolve()
        self.parent_guard = Path(parent_guard).resolve()
        self.read, self.sha = read, sha

    def __call__(self, *, before_start):
        project = self.read(self.project)
        goal, task = project["goal_execution"], project["staged_training_request"]
        if (goal.get("status") != "active" or goal.get("pause_requested_by_user") is not False
                or task.get("pause_requested") is not False or task.get("current_stage") != self.stage
                or task.get("current_protocol_frozen") is not True
                or task.get("current_protocol") != self.protocol_name):
            return False
        if before_start and (task.get("learning_running") is not False or task.get("simulation_running") is not False):
            return False
        raw = self.protocol.read_bytes()
        plan = json.loads(raw)
        return (hashlib.sha256(raw).hexdigest() == self.digest and isinstance(plan, dict)
                and isinstance(plan.get("supervisor_workers"), dict)
                and plan["supervisor_workers"].get(self.run_id) == self.command
                and plan.get("supervisor_sha256") == self.sha(self.supervisor_source)
                and plan.get("parent_exit_guard_sha256") == self.sha(self.parent_guard))
