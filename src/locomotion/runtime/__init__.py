"""Run an exported walking policy on the physical robot: numpy only, no simulator or torch.

``export.py`` (the one module here that needs torch) turns a ``terrain_v2_odom`` checkpoint into
a bundle; ``bundle.py`` loads and verifies it; ``inputs.py`` rebuilds the 278 actor inputs from
real sensors with leg odometry; ``controller.py`` runs the 50 Hz loop with its safety states;
``hardware.py`` maps motors and the IMU onto the canonical joints and body frame.
"""
