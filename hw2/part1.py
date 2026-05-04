"""
AA 276 Homework 1 | Coding Portion | Part 1 of 3


OVERVIEW

In this file, you will implement several functions required by the 
neural CBF library developed by the REALM Lab at MIT to
automatically learn your own CBFs for a 13D quadrotor system!

From this exercise, you will hopefully better understand the course
materials through a concrete example, appreciate the advantages
(and disadvantages) of learning a CBF versus manually constructing
one, and get some hands-on coding experience with using state-of-the-art
tools for synthesizing safety certificates, which you might find
useful for your own work.

If you are interested in learning more, you can find the library
here: https://github.com/MIT-REALM/neural_clbf


INSTRUCTIONS

Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check1.py`.
After the tests pass, train a neural CBF (in your VM) by running `python scripts/train.py`.


IMPORTANT NOTES ON TRAINING
The training can take a substantial amount of time to complete [~9 hours ~= $10].
However, you should be able to implement all code for Parts 1, 2, and 3 in the meantime.
After each training epoch [50 total], the CBF model will save to 'outputs/cbf.ckpt'.
As long as you have at least one checkpoint saved [~10 minutes], Part 3 will load this checkpoint.
Try your best to not exceed $10 in credits -  you can stop training early if you reach this budget limit.

REMEMBER TO SHUTDOWN YOUR VIRTUAL MACHINES AFTER TRAINING, TO AVOID ACCUMULATING FEES.
"""

import torch

def state_limits():
    """
    Return a tuple (upper, lower) describing the state bounds for the system.
    Order: (px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz)
    """
    # YOUR CODE HERE
    upper = torch.tensor([3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5], dtype=torch.float32)
    lower = torch.tensor([-3, -3, -3, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5], dtype=torch.float32)
    return (upper, lower)


def control_limits():
    """
    Return a tuple (upper, lower) describing the control bounds for the system.
    Order: (F, alpha_x, alpha_y, alpha_z)
    """
    # YOUR CODE HERE
    upper = torch.tensor([20, 8, 8, 4], dtype=torch.float32)
    lower = torch.tensor([-20, -8, -8, -4], dtype=torch.float32)
    return (upper, lower)

def safe_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the prescribed safe set.
    Safe set C = {x : sqrt(px^2 + py^2) > 2.8}
    """
    # YOUR CODE HERE
    px, py = x[:, 0], x[:, 1]
    dist = torch.sqrt(px**2 + py**2)
    return dist > 2.8


def failure_mask(x):
    """
    Return a boolean tensor indicating whether the states x are in the failure set.
    Failure set L = {x : sqrt(px^2 + py^2) < 0.5}
    """
    # YOUR CODE HERE
    px, py = x[:, 0], x[:, 1]
    dist = torch.sqrt(px**2 + py**2)
    return dist < 0.5


def f(x):
    """
    Return the control-independent part of the control-affine dynamics.
    Derived from the provided differential equations.
    """
    PXi, PYi, PZi, QWi, QXi, QYi, QZi, VXi, VYi, VZi, WXi, WYi, WZi = [i for i in range(13)]
    PX, PY, PZ, QW, QX, QY, QZ, VX, VY, VZ, WX, WY, WZ = [x[:, i] for i in range(13)]

    f = torch.zeros_like(x)
    # Position dynamics: dot_p = v
    f[:, PXi] = VX
    f[:, PYi] = VY
    f[:, PZi] = VZ
    
    # Quaternion dynamics
    f[:, QWi] = -0.5 * (WX * QX + WY * QY + WZ * QZ)
    f[:, QXi] =  0.5 * (WX * QW + WZ * QY - WY * QZ)
    f[:, QYi] =  0.5 * (WY * QW - WZ * QX + WX * QZ)
    f[:, QZi] =  0.5 * (WZ * QW + WY * QX - WX * QY)
    
    # Linear velocity drift: Gravity only in Z-direction
    f[:, VZi] = -9.8
    
    # Angular velocity drift (gyroscopic terms)
    f[:, WXi] = -5.0 * WY * WZ / 9.0
    f[:, WYi] =  5.0 * WX * WZ / 9.0
    # dot_wz has no drift term (purely alpha_z)
    
    return f

def g(x):
    """
    Return the control-dependent part of the control-affine dynamics.
    Derived from the provided differential equations.
    """
    # YOUR CODE HERE
    batch_size = x.shape[0]
    qw, qx, qy, qz = x[:, 3], x[:, 4], x[:, 5], x[:, 6]
    
    # Initialize g matrix: shape [batch_size, 13 states, 4 controls]
    g = torch.zeros((batch_size, 13, 4), device=x.device, dtype=x.dtype)
    
    # Thrust F column (u index 0)
    # v_dot_x = 2F(qw*qy + qx*qz)
    # v_dot_y = 2F(qy*qz - qw*qx)
    # v_dot_z = 2F(0.5 - qx^2 - qy^2) - 9.8 (drift term is in f)
    g[:, 7, 0] = 2 * (qw * qy + qx * qz)
    g[:, 8, 0] = 2 * (qy * qz - qw * qx)
    g[:, 9, 0] = 2 * (0.5 - qx**2 - qy**2)
    
    # Angular Acceleration columns (u indices 1, 2, 3)
    # w_dot_x = alpha_x - 5/9*wy*wz
    # w_dot_y = alpha_y + 5/9*wx*wz
    # w_dot_z = alpha_z
    g[:, 10, 1] = 1.0
    g[:, 11, 2] = 1.0
    g[:, 12, 3] = 1.0
    
    return g