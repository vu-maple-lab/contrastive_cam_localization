import torch 

def loss_fn(loss, batch_size, z_anchor, z_positive, z_negatives, cam_diff, device, alpha=0.1, b=0.1): 
    first_weight_term = float(alpha) * (float(cam_diff[:,0].item()) / 5)
    res = first_weight_term * loss(z_anchor, z_positive, target=torch.ones((batch_size,)).to(device))
    for z_negative in z_negatives:
        res += b * loss(z_anchor, z_negative, target=-torch.ones((batch_size,)).to(device))
    return res