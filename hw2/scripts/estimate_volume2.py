import torch
from problem4_helper import NeuralVF, NeuralCBF

def estimate_volume(model_type='vf', n_samples=100000):
    # 1. Initialize the helper
    if model_type == 'vf':
        helper = NeuralVF(ckpt_path='outputs/vf.ckpt')
    else:
        helper = NeuralCBF(ckpt_path='outputs/cbf.ckpt')

    # 2. Define State Space Bounds (example for Quadrotor 13D)
    # Adjust these bounds based on the problem statement specs!
    # Usually [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
    # This is a dummy range; check your HW PDF for the exact state range.
    mins = torch.tensor([-5.0, -5.0, 0.0, -1.0, -1.0, -3.14, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, 0.0])
    maxs = torch.tensor([ 5.0,  5.0, 10.0,  1.0,  1.0,  3.14,  2.0,  2.0,  2.0,  1.0,  1.0,  1.0, 1.0])

    # 3. Generate Random Samples
    # x = low + (high - low) * random_tensor
    samples = mins + (maxs - mins) * torch.rand((n_samples, 13))

    # 4. Query Values
    # Since n_samples might be large, we batch them to avoid OOM
    batch_size = 10000
    safe_count = 0
    
    for i in range(0, n_samples, batch_size):
        batch = samples[i : i + batch_size]
        vals = helper.values(batch)
        
        # Check safety (V >= 0 or h >= 0)
        safe_count += torch.sum(vals >= 0).item()

    proportion = safe_count / n_samples
    print(f"Model: {model_type.upper()}")
    print(f"Estimated Volume Proportion: {proportion:.4f}")
    return proportion

if __name__ == "__main__":
    # Uncomment the one you are currently set up to run:
    # estimate_volume(model_type='vf')
    estimate_volume(model_type='cbf')