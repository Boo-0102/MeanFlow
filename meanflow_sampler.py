import torch
import torch.nn.functional as F

@torch.no_grad()
def meanflow_sampler(
    model, 
    latents,
    num_steps=1,
    do_cfg=False,
    num_classes=10,
    class_label=None,
    **kwargs
):
    """
    MeanFlow sampler supporting both single-step and multi-step generation
    """
    batch_size = latents.shape[0]
    device = latents.device
    # Use CFG
    if do_cfg:
        labels = torch.full((batch_size,), class_label, device=device) # [b,]
        c = F.one_hot(labels, num_classes=num_classes).float() # [b,] -> [b, 10]

        if num_steps == 1:
            r = torch.zeros(batch_size, device=device)
            t = torch.ones(batch_size, device=device)
            u = model(latents, noise_labels_r=r, noise_labels_t=t, class_labels=c)
            # x_0 = x_1 - u(x_1, 0, 1)
            x0 = latents - u
            
        else:
            z = latents
            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]
                
                t = torch.full((batch_size,), t_cur, device=device)
                r = torch.full((batch_size,), t_next, device=device)

                u = model(z, noise_labels_r=r, noise_labels_t=t, class_labels=c)
                
                # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
                z = z - (t_cur - t_next) * u
            
            x0 = z
    # No CFG
    else:    
        if num_steps == 1:
            r = torch.zeros(batch_size, device=device)
            t = torch.ones(batch_size, device=device)
            u = model(latents, noise_labels_r=r, noise_labels_t=t)
            # x_0 = x_1 - u(x_1, 0, 1)
            x0 = latents - u
            
        else:
            z = latents
            time_steps = torch.linspace(1, 0, num_steps + 1, device=device)
            for i in range(num_steps):
                t_cur = time_steps[i]
                t_next = time_steps[i + 1]
                
                t = torch.full((batch_size,), t_cur, device=device)
                r = torch.full((batch_size,), t_next, device=device)

                u = model(z, noise_labels_r=r, noise_labels_t=t)
                
                # Update z: z_r = z_t - (t-r)*u(z_t, r, t)
                z = z - (t_cur - t_next) * u
            
            x0 = z
    
    return x0