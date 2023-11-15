import torch 

def loss_fn(loss, batch_size, z_anchor, z_positive, z_negatives, device, b=0.1): 
    
    res = loss(z_anchor, z_positive, target=torch.ones((batch_size,)).to(device))
    for z_negative in z_negatives:
        res += b * loss(z_anchor, z_negative, target=-torch.ones((batch_size,)).to(device))
    return res