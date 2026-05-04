import torch
from problem4_helper import NeuralVF, NeuralCBF
def estimate_volume(model_type='vf', total_samples=500000, batch_size=10000):
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
    
    safe_count = 0
    
    # Process in batches to avoid the "Killed" error
    num_batches = total_samples // batch_size
    
    for i in range(num_batches):
        # Generate samples ONLY for this batch
        samples = mins + (maxs - mins) * torch.rand((batch_size, 13))
        
        # Move to GPU if available, or stay on CPU
        # samples = samples.cuda() 
        
        with torch.no_grad():
            vals = helper.values(samples)
            safe_count += torch.sum(vals >= 0).item()
            
        # Optional: Print progress so you know it's not frozen
        if i % 10 == 0:
            print(f"Batch {i}/{num_batches} complete...")

    proportion = safe_count / (num_batches * batch_size)
    print(f"Final Proportion: {proportion}")   
    return proportion

if __name__ == "__main__":
    # Uncomment the one you are currently set up to run:
    # estimate_volume(model_type='vf')
    estimate_volume(model_type='cbf')