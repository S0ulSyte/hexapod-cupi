"""Calibrate the simulator's leg actuation to the single-leg stand (system identification).

``leg_record.py`` runs on the stand computer and records a scripted excitation; ``leg_sim.py`` replays the
same commands on a one-leg MuJoCo model built from the approved robot assets; ``leg_fit.py`` fits the
joint mapping, actuation delay and actuator parameters so the replay matches the recording.
"""
