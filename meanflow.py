from functools import partial

import torch
import torch.func
import torch.nn.functional as F

from utils.loss import compute_loss


class MeanFlow:
    def __init__(
            self,
            path_type="linear",
            weighting="uniform",
            # New parameters
            time_sampler="logit_normal",  # Time sampling strategy: "uniform" or "logit_normal"
            time_mu=-2.0,                 # Mean parameter for logit_normal distribution
            time_sigma=2.0,               # Std parameter for logit_normal distribution
            ratio_r_not_equal_t=0.25,     # Ratio of samples where r≠t
            adaptive_p=1.0,               # Power param for adaptive weighting
            label_dropout_prob=0.1,       # Drop out label
            # CFG related params
            do_cfg = False,               # Use CFG or not 
            cfg_omega=1.0,                # CFG omega param, default 1.0 means no CFG
            cfg_kappa=0.0,
            num_classes=10
            ):
        self.weighting = weighting
        self.path_type = path_type
        
        # Time sampling config
        self.time_sampler = time_sampler
        self.time_mu = time_mu
        self.time_sigma = time_sigma
        self.ratio_r_not_equal_t = ratio_r_not_equal_t
        self.label_dropout_prob = label_dropout_prob
        # Adaptive weight config
        self.adaptive_p = adaptive_p

         # CFG config
        self.do_cfg = do_cfg
        self.cfg_omega = cfg_omega
        self.cfg_kappa = cfg_kappa
        self.num_classes = num_classes

    def interpolant(self, t):
        """Define interpolation function
            z_t = alpha_t * x + sigma_t * ε
            v_t = d_alpha_t * x + d_sigma_t * ε
        
        """
        if self.path_type == "linear":
            # z_t = (1-t) * x + t * ε  vt = ε - x
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        else:
            raise NotImplementedError()
        
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t
    
    def sample_time_steps(self, batch_size, device):
        """Sample time steps (r, t) according to the configured sampler

            - uniform: a uniform distribution, U(0, 1)
            - logit_normal: a sample is first drawn from a normal distribution N (μ, σ),
                          then mapped to (0, 1) using the logistic function
        """
        # Step1: Sample two time points
        if self.time_sampler == "uniform":
            time_samples = torch.rand(batch_size, 2, device=device)
        elif self.time_sampler == "logit_normal":
            normal_samples = torch.randn(batch_size, 2, device=device) # N(0,1)
            normal_samples = normal_samples * self.time_sigma + self.time_mu # N(-2,2)
            time_samples = torch.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unknown time sampler: {self.time_sampler}")
        
        # Step2: Ensure t > r by sorting
        sorted_samples, _ = torch.sort(time_samples, dim=1)
        r, t = sorted_samples[:, 0], sorted_samples[:, 1]
        
        # Step3: Control the proportion of r=t samples
        fraction_equal = 1.0 - self.ratio_r_not_equal_t  # e.g., 0.75 means 75% of samples have r=t
        # Create a mask for samples where r should equal t
        equal_mask = torch.rand(batch_size, device=device) < fraction_equal
        # Apply the mask: where equal_mask is True, set r=t (replace)
        r = torch.where(equal_mask, t, r)
        
        return r, t

    def __call__(self, model, images, labels):
        """
        Compute MeanFlow loss function (unconditional)
        """
        batch_size = images.shape[0] # imgaes:[b,c,32,32]
        device = images.device

        # Sample time steps
        r, t = self.sample_time_steps(batch_size, device) # [b]
        t_ = t.view(-1, 1, 1, 1) # [b, 1, 1, 1]
        r_ = r.view(-1, 1, 1, 1) # [b, 1, 1, 1]

        # Sample noise
        noises = torch.randn_like(images)
        
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t_)

        """
        z_t = alpha_t * x + sigma_t * ε    v_t = d_alpha_t * x + d_sigma_t * ε
        - if Linear interpolation:z_t = (1-t) * x + t * ε   vt = ε - x   
        """
        # Get Z_t V_t
        z_t = alpha_t * images + sigma_t * noises
        v_t = d_alpha_t * images + d_sigma_t * noises
        # - 在单GPU下，会直接使用 model, 在多GPU下，会获取 model.module
        unet_model = model.module if hasattr(model, 'module') else model
        
        ## Using CFG
        if self.do_cfg:
            # one-hot
            c = F.one_hot(labels, num_classes=self.num_classes).float() # [b,] -> [b, 10]

            # Get v_hat
            v_hat = self.cfg_omega * v_t + self.cfg_kappa * unet_model(z_t, t, t, c) + \
                    (1 - self.cfg_kappa - self.cfg_omega) * unet_model(z_t, t, t)
            
            # Get Jacobian matrix
            model_partial = partial(unet_model, class_labels=c)
            _, dudt = torch.func.jvp(
                model_partial,
                (z_t, r, t),  
                (v_hat, torch.zeros_like(r), torch.ones_like(t)) 
            )

            # Get error
            u_pred = unet_model(z_t, r, t, c)
            u_target = v_hat - (t_ - r_) * dudt # utgt = vt − (t − r) * (vt * ∂_z u + ∂_t u)
            error = u_pred - u_target.detach()
            
        ## No CFG
        else:
            # Get Jacobian matrix
            _, dudt = torch.func.jvp(
                unet_model,
                (z_t, r, t),  
                (v_t, torch.zeros_like(r), torch.ones_like(t)) 
            )

            # Get error
            u_pred = unet_model(z_t, r, t)
            u_target = v_t - (t_ - r_) * dudt
            error = u_pred - u_target.detach()
        
        # Compute loss
        loss = compute_loss(error=error, weighting=self.weighting, adaptive_p=self.adaptive_p)
        
        return loss
