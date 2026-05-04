"""
AA 276 Homework 1 | Coding Portion | Part 2 of 3


OVERVIEW

In this file, you will implement functions for simulating the
13D quadrotor system discretely and computing the CBF-QP controller.


INSTRUCTIONS

Make sure you pass the tests for Part 1 before you begin.
Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check2.py`.
"""


import torch
from part1 import f, g


"""Note: the following functions operate on batched inputs."""

def euler_step(x, u, dt):
    """
    Return the next states xn using a discrete Euler step.
    """
    # Dynamics: x_dot = f(x) + g(x) @ u
    # Note: torch.bmm performs batch matrix multiplication
    x_dot = f(x) + torch.bmm(g(x), u.unsqueeze(-1)).squeeze(-1)
    
    # Equation (3): x_{t+1} = x_t + dt * x_dot
    xn = x + dt * x_dot
    return xn

    
def roll_out(x0, u_fn, nt, dt):
    """
    Return the state trajectories starting with x1.
    """
    xts = []
    x_curr = x0
    for _ in range(nt):
        u = u_fn(x_curr)
        x_curr = euler_step(x_curr, u, dt)
        xts.append(x_curr)
        
    # Stack along the time dimension [batch_size, nt, 13]
    return torch.stack(xts, dim=1)

import cvxpy as cp
import numpy as np
from part1 import control_limits

def u_qp(x, h, dhdx, u_ref, gamma, lmbda):
    """
    Solve the CBF-QP optimization.
    """
    batch_size = x.shape[0]
    u_upper, u_lower = control_limits()
    u_upper_np = u_upper.detach().cpu().numpy()
    u_lower_np = u_lower.detach().cpu().numpy()
    
    # Pre-calculate Lie derivatives for efficiency
    f_val = f(x) # [batch_size, 13]
    g_val = g(x) # [batch_size, 13, 4]
    
    # Lf h = (dh/dx) * f(x)
    Lfh = (dhdx * f_val).sum(dim=-1).detach().cpu().numpy()
    # Lg h = (dh/dx) * g(x)
    Lgh = torch.bmm(dhdx.unsqueeze(1), g_val).squeeze(1).detach().cpu().numpy()
    
    h_np = h.detach().cpu().numpy()
    u_ref_np = u_ref.detach().cpu().numpy()
    
    u_results = []
    
    # Loop through batch as CVXPY 1.2 is not natively batched for GPUs
    for i in range(batch_size):
        u_var = cp.Variable(4)
        delta_var = cp.Variable(1, nonneg=True) # delta >= 0
        
        # Equation (4): Minimize ||u - u_ref||^2 + lambda * delta^2
        obj = cp.Minimize(cp.sum_squares(u_var - u_ref_np[i]) + lmbda * cp.square(delta_var))
        
        # Equation (5-7): Safety and Control Constraints
        constraints = [
            Lfh[i] + Lgh[i] @ u_var + gamma * h_np[i] + delta_var >= 0,
            u_var >= u_lower_np,
            u_var <= u_upper_np
        ]
        
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, verbose=False)
        
        # Handle cases where the solver might fail by defaulting to u_ref
        if u_var.value is None:
            u_results.append(u_ref[i].cpu())
        else:
            u_results.append(torch.from_numpy(u_var.value).float())
            
    return torch.stack(u_results).to(x.device)