"""
AA 276 Homework 1 | Coding Portion | Part 3 of 3


OVERVIEW

In this file, you will implement functions for 
visualizing your learned CBF from Part 1 and evaluating
the accuracy of the learned CBF and corresponding CBF-QP controller.


INSTRUCTIONS

Make sure you pass the tests for Part 1 and Part 2 before you begin.
Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, make sure that (in your VM) there is a CBF model checkpoint
saved at `outputs/cbf.ckpt`. Then, run `python scripts/plot.py`.
Submit the false safety rate reported in the terminal and the plot that is
saved to `outputs/plot.png`.

REMEMBER TO SHUTDOWN YOUR VIRTUAL MACHINES AFTER TRAINING, TO AVOID ACCUMULATING FEES.
"""


import torch

import matplotlib.pyplot as plt

def plot_h(fig, ax, px, py, slice, h_fn):
    # Remove the indexing='ij' argument for compatibility with the VM's PyTorch version
    PX, PY = torch.meshgrid(px, py) 
    
    # The rest of the setup remains the same
    X = torch.zeros((len(px), len(py), 13))
    X[..., 0] = PX
    X[..., 1] = PY
    X[..., 2:] = slice[2:]
    
    # Reshape for the batch-processing h_fn
    X_flat = X.reshape(-1, 13)
    with torch.no_grad():
        h_vals = h_fn(X_flat).reshape(len(px), len(py))
        
    # Plotting using the grid defined by px and py
    c = ax.pcolormesh(px.numpy(), py.numpy(), h_vals.numpy().T, cmap='RdYlGn', shading='auto')
    fig.colorbar(c, ax=ax, label='Learned $h(x)$')
    
    # Zero level set contour to indicate the safety boundary
    ax.contour(px.numpy(), py.numpy(), h_vals.numpy().T, levels=[0], colors='k', linewidths=2)
    
from part1 import safe_mask, failure_mask
from part2 import roll_out, u_qp

def plot_and_eval_xts(fig, ax, x0, u_ref_fn, h_fn, dhdx_fn, gamma, lmbda, nt, dt):
    # Define the safety-filtered controller
    def u_fn(x):
        return u_qp(x, h_fn(x), dhdx_fn(x), u_ref_fn(x), gamma, lmbda)
        
    # Simulate trajectories starting from x0
    xts = roll_out(x0, u_fn, nt, dt) # Shape: [batch_size, nt, 13]
    
    # Plot projected trajectories on the 2D position space (px, py)
    for i in range(x0.shape[0]):
        ax.plot(xts[i, :, 0].detach().cpu().numpy(), 
                xts[i, :, 1].detach().cpu().numpy(), 
                color='k', alpha=0.2)
        
    # Calculate False Safety Rate
    is_initial_safe = safe_mask(x0) # States identified as safe by Part 1 definition
    
    # Check if any point in each trajectory enters the failure set
    batch_size = x0.shape[0]
    xts_flat = xts.reshape(-1, 13)
    is_failure_flat = failure_mask(xts_flat) 
    is_failure_any = is_failure_flat.reshape(batch_size, nt).any(dim=1) 
    
    # Trajectories that started safe but resulted in a failure
    false_safe_mask = is_initial_safe & is_failure_any
    
    num_initial_safe = is_initial_safe.sum().item()
    if num_initial_safe == 0:
        return 0.0
        
    false_safety_rate = false_safe_mask.sum().item() / num_initial_safe
    
    return false_safety_rate