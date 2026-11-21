"""Verify the computed early-load output before recovering its project receipt."""

from unittest.mock import patch

import review_nominal_low_friction_early_left_turn_load as audit
from review_nominal_low_friction_failure_recurrence import permitted, read, rel, sha, write


def main():
    project = read(audit.PROJECT)
    permitted(project,audit.STAGE)
    node = project['goal_execution'][audit.KEY]
    assert node['status']=='archive-ready' and audit.OUTPUT.exists()
    assert sha(audit.PLAN)==node['protocol_sha256']
    result = read(audit.OUTPUT)
    assert result['status']=='verified-development-load-readback'
    assert not any(result[k] for k in ('additional_physical_steps','learning_updates','low_friction_reserved_seed_used','hardware_io'))
    field = result['first_support_shift_window_ms']
    node.update(status='reviewed',result=rel(audit.OUTPUT),result_sha256=sha(audit.OUTPUT),
        earliest_support_shift_window_ms=field,report=rel(audit.REPORT/'README.md'))
    info = dict(current_protocol_execution=audit.STAGE+'-reviewed',current_evidence=rel(audit.OUTPUT))
    project.update(info)
    project['staged_training_request'].update(info,next_gate=
        'Verify actual force and motor changes before selecting a matched early transfer intervention.' if field else
        'Compare early pitch and head unloading without a left-right load-shift assumption, then select the next preflight.')

    def read_pending(path, expected=None):
        return project if path==audit.PROJECT else read(path,expected)

    with patch.object(audit,'read',read_pending):
        audit.score(True)
    write(audit.PROJECT,project)
    audit.score(True)
    print('EARLY_LEFT_TURN_PROJECT_RECEIPT_RECOVERED',node['result_sha256'],flush=True)


if __name__=='__main__':
    main()
