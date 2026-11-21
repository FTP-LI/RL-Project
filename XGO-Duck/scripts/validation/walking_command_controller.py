"""Portable fixed controller selected by the closed turn/strafe screen.

Original ONNX and raw action history stay unchanged. The inherited controller
uses only requested commands and the actor's existing gyro, keeps resettable
EMA state, and leaves zero-turn inputs identical to the fixed mapping.
Reliability checks are in the frozen turn-feedback preflight and replay.
"""
from gyro_turn_controller import Controller


def create_controller(num_envs,dt=.02):
    return Controller(['gyro-feedback']*num_envs,dt)
