#!/usr/bin/env bash
# One shared visible VS Code terminal; no daemon, background training or install.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$HOME/XGO-Duck/workspace/xgoduck_rl"
PY="$WORK/.venv/bin/python"
export PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
cd "$WORK"
case "${1:-status}" in
  roll-endpoint-confirmation)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_endpoint_confirmation.py" ;;
  roll-endpoint-readback)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_endpoint_readback.py" ;;
  roll-contact-clock)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_contact_clock.py" ;;
  getup-recovery-retention)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_recovery_retention_pair.py" ;;
  getup-supine-weighted-learning)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_supine_training_pair.py" ;;
  getup-anchor-same-release)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_anchor_same_release.py" ;;
  roll-neck-support-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_neck_support_log_recovery.py" ;;
  roll-neck-support)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_neck_support_trial.py" ;;
  walk-forward-phase)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/common_walk_forward_phase_screen.py" ;;
  getup-contact-shift)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_contact_shift.py" ;;
  getup-knee-trigger-frames)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/review_getup_knee_screen_motion.py" --trigger-frames ;;
  getup-knee-support-supply-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_knee_support_supply_recovery.py" ;;
  getup-knee-support-screen)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_knee_support_screen.py" ;;
  getup-shared-release-terrain)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_shared_release_terrain.py" ;;
  getup-shared-release)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_shared_release.py" ;;
  getup-persistent-contact-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_persistent_contact_pair.py" ;;
  getup-persistent-value)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/getup_persistent_value.py" ;;
  common-walk-resets)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/recover_common_walk_resets.py" ;;
  common-walk-confirm)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/common_walk_confirmation.py" ;;
  getup-teacher-anchor-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_teacher_anchor_pair.py" ;;
  getup-anchor-scale-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/recover_getup_anchor_scale.py" ;;
  getup-anchor-value-check)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/getup_anchor_value_check.py" ;;
  roll-rng-confirm)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/recover_roll_import_rng.py" ;;
  roll-common-confirm)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_common_confirmation.py" ;;
  getup-early-guard-replay)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/render_getup_early_guard_replay.py" ;;
  getup-early-guard)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_early_guard_study.py" ;;
  getup-late-feasibility-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_late_feasibility_pair.py" ;;
  getup-interface-value-readback)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/recover_getup_interface_value.py" ;;
  getup-interface-value)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/getup_interface_value.py" ;;
  getup-neck-home-order)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_neck_home_order_study.py" ;;
  getup-stand-handoff-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_stand_handoff_pair.py" ;;
  getup-relief-state-replay)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/render_getup_relief_replay.py" ;;
  getup-neck-relief-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_neck_relief_pair.py" ;;
  getup-release-coverage)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_release_coverage.py" ;;
  getup-late-contact-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_contact_pair.py" ;;
  complete-getup-replay)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/render_complete_getup.py" ;;
  getup-value-bootstrap-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/recover_getup_value_startup.py" ;;
  getup-value-bootstrap)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/getup_value_bootstrap.py" ;;
  getup-contact-state-replay)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/render_getup_contact_replay.py" ;;
  getup-contact-release-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/recover_getup_contact_release.py" ;;
  getup-contact-isolation)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/getup_contact_isolation.py" ;;
  common-plant-screen-freeze)
    exec "$PY" -B "$ROOT/scripts/validation/common_plant_screen.py" --freeze ;;
  common-plant-screen)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/common_plant_screen.py" ;;
  common-plant-audit)
    exec "$PY" -B "$ROOT/scripts/validation/inspect_common_skill_plant.py" ;;
  roll-angle-recovery-freeze)
    exec "$PY" -B "$ROOT/scripts/validation/recover_angle_feature_pair.py" --freeze ;;
  roll-angle-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/recover_angle_feature_pair.py" ;;
  roll-angle-freeze)
    exec "$PY" -B "$ROOT/scripts/validation/angle_feature_pair.py" --freeze ;;
  roll-angle-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/angle_feature_pair.py" ;;
  roll-angle-input-audit)
    exec "$PY" -B "$ROOT/scripts/validation/inspect_roll_angle_inputs.py" ;;
  roll-stand-preparation)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/stand_preparation_pair.py" ;;
  roll-completion-frontier-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/recover_completion_frontier.py" ;;
  roll-completion-frontier)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/completion_frontier_pair.py" ;;
  roll-adaptation-screen)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_model_screen.py" ;;
  roll-top-contact-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_top_contact_pair.py" ;;
  roll-action-memory-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/recover_roll_action_memory.py" ;;
  roll-action-memory-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/roll_action_memory_pair.py" ;;
  practice-walking)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/practice_walking.py" --name "${2:-walk-practice}" --seed "${3:-1101}" ;;
  seven-command-confirmation)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/seven_command_confirmation.py" ;;
  turn-strafe-screen)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/turn_feedback_probe.py" ;;
  straight-parallel-confirmation)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/batched_velocity_confirmation.py" ;;
  command-mapping-probe)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/command_mapping_probe.py" ;;
  command-response-probe)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/command_response_probe.py" ;;
  planar-filter-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/run_planar_filter_pair.py" ;;
  roll-reset-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/evaluate_roll_reset_pair.py" ;;
  roll-reference)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/evaluate_roll_reference.py" ;;
  random-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/evaluate_random_recovery.py" ;;
  lateral-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/evaluate_lateral_recovery.py" ;;
  standup-reference)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/evaluate_standup_reference.py" ;;
  status) exec "$PY" -B "$ROOT/scripts/validation/current_status.py" ;;
  verify) exec "$PY" "$ROOT/scripts/validation/verify_training_workflow.py" ;;
  preflight) exec "$PY" "$ROOT/scripts/training/guarded_training.py" preflight ;;
  reference-inputs)
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/s1_reference_inputs.py" ;;
  reference-evaluation)
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/s1_reference_evaluation.py" ;;
  report) "$PY" "$ROOT/scripts/validation/render_development_report.py"; exec "$PY" "$ROOT/scripts/validation/verify_report_artifacts.py" ;;
  hip-repair)
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/run_hip_pair.py" ;;
  official-bootstrap)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/official_bootstrap.py" ;;
  official-bootstrap-screen)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/evaluate_official_bootstrap.py" ;;
  official-continuation)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/official_continuation.py" ;;
  official-adaptation-recovery)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/training/resume_official_adaptation.py" ;;
  official-adaptation-screen)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/evaluate_adaptation_screen.py" ;;
  tracking-precision-pair)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/run_tracking_precision_pair.py" ;;
  reward-incentive-audit)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/reward_incentive_audit.py" ;;
  observer-integration)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" -B "$ROOT/scripts/validation/s1_observer_integration.py" ;;
  run)
    exec 9>"$HOME/XGO-Duck/training.lock"
    flock -n 9 || { echo 'Another project compute task is active.'; exit 3; }
    source "$ROOT/scripts/setup/project_environment.sh"
    export MUJOCO_GL=egl GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
    exec "$PY" "$ROOT/scripts/training/guarded_training.py" run ;;
  *) echo 'Unknown workflow action'; exit 2 ;;
esac
