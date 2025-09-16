from typing import Dict, List
import warnings
from tqdm import tqdm

from einops import rearrange
import torch
from torch.distributions import LogisticNormal

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py
def mean_flat(tensor: torch.Tensor, eps=1e-6):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_transform(
    t,
    scale=1.0,
    num_timesteps=1,
):
    new_t = scale * t / (1 + (scale - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from javisdit/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            t = self.sample_timestep(x_start)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)

        terms = {}
        velocity_pred = model(x_t, t, **model_kwargs)
        loss = (velocity_pred - (x_start - noise)).pow(2)
        loss = loss.mean()
        terms["loss"] = loss
        return terms

    def sample_timestep(self, x_start):
        if self.use_discrete_timesteps:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        elif self.sample_method == "uniform":
            t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
        elif self.sample_method == "logit-normal":
            t = self.sample_t(x_start) * self.num_timesteps
        if self.use_timestep_transform:
            t = timestep_transform(t, scale=self.transform_scale, num_timesteps=self.num_timesteps)
        return t

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        target_dim = noise.shape[1:]
        timepoints = timepoints.view(-1, *([1] * len(target_dim)))
        timepoints = timepoints.repeat(1, *target_dim)

        return timepoints * original_samples + (1 - timepoints) * noise


class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        z, # noise
        model_args,
        device,
        additional_args=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        # prepare timesteps
        ## num_timesteps means diffusion/training steps; num_sampling_steps means denoising/inference steps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in enumerate(progress_wrap(timesteps)):
            # classifier-free guidance
            z_in = torch.cat([z, z], 0) # [bs * 2, c, d, h, w]
            t = torch.cat([t, t], 0) # [bs * 2]
            pred = model.forward_with_cfg(z_in, t, **model_args)
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]
        return z
    
    def training_losses(self, model, x_start, model_kwargs=None, noise=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, t)