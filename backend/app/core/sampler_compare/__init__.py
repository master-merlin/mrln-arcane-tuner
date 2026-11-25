"""A held-out comparison of timestep samplers on one family and dataset.

The runner stays torch-free: product facts that need the engine are asked of a child process
(``facts``), and every outside reading (card, RAM, clock, the two child command lines) goes
through one ``runner.Seams`` object.
"""
