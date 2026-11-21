"""Use the verified three-command evaluator for the reviewed adaptation."""
import evaluate_official_bootstrap as entry
entry.PLAN=entry.ROOT/'configs/experiments/official_adaptation_screen.json'
entry.OUT=entry.ROOT/'records/reports/official-adaptation'
if __name__=='__main__':entry.main()
