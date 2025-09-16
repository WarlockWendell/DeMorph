import numpy as np
from generative.metrics import MultiScaleSSIMMetric
import torch


def cal_psnr_ssim(img1, img2, max_val=1.0, window_size=11, sigma=1.5):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        psnr_valur = float('inf')
    else:
        psnr_valur = 20 * np.log10(max_val / np.sqrt(mse))
    
    ms_ssim = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=5)
    ssim_val = ms_ssim(torch.from_numpy(img1[None, None]), torch.from_numpy(img2[None, None]))

    return psnr_valur, ssim_val