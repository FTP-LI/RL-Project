"""Recompute outcome-linked kinematics for the replay-coverage comparison."""
from unittest.mock import patch

from evaluate_getup_side_to_supine_pair import OUT,PLAN,ROLES
import review_getup_early_support_ppo_effect as effect


if __name__=='__main__':
    with patch.object(effect,'OUT',OUT),patch.object(effect,'PLAN',PLAN),patch.object(effect,'ROLES',ROLES):
        effect.main()
