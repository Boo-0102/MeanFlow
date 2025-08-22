import torch

def compute_loss(error, weighting, adaptive_p):
    """
    Loss function: Adaptive-L2 loss or L2 loss 
    """
    if weighting == "adaptive":
        error_norm = torch.norm(error.reshape(error.shape[0], -1), dim=1)
        weights = 1.0 / (error_norm.detach() ** 2 + 1e-3).pow(adaptive_p)
        loss = weights * error_norm ** 2
    else:
        error_norm = torch.norm(error.reshape(error.shape[0], -1), dim=1) # [b,3,32,32]-> [b,2*32*32] -> [b,]
        loss = error_norm ** 2
    
    return loss # [b,]
