import os
import os.path as osp
import argparse
import yaml
import numpy as np
import torch
import pandas as pd
import nibabel as nib
from monai import transforms
from nibabel.processing import resample_from_to
import pickle
import utils.const as const
from generative.networks.nets import AutoencoderKL
from utils.rflow import RFLOW
from models.build import build_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../config/ode.yaml')
    parser.add_argument('--output', type=str, default='../output')
    parser.add_argument('--weights', type=str, default='../weights/checkpoint.pth')
    parser.add_argument('--data_meta', type=str, default='../example/meta.yaml')
    parser.add_argument('--vis', type=bool, default=False)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg['model'])
    model.load_state_dict(torch.load(args.weights, weights_only=False, map_location='cpu')['model_ema'])
    model = model.cuda().eval()

    data = yaml.safe_load(open(args.data_meta, 'r'))
    current_image_path = data['current_image_path']
    current_segm_path = data['current_segm_path']
    current_age = data['current_age']
    target_age = data['target_age']
    current_dx = data['current_dx']
    target_dx = data['target_dx']
    current_mmse = data['current_mmse']
    sex = data['sex']

    load_volume_tensor = transforms.Compose([
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image'])
    ])
    load_segm_tensor = transforms.Compose([
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']), 
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image'], mode='nearest'),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
    ])

    current_image = load_volume_tensor({'image': current_image_path})['image'].unsqueeze(0)
    current_segm = load_segm_tensor({'image': current_segm_path})['image'].unsqueeze(0)

    sex = 0 if sex == 'Male' else 1
    current_dx = 0 if current_dx == 'CN' else 1 if current_dx == 'MCI' else 2
    target_dx = 0 if target_dx == 'CN' else 1 if target_dx == 'MCI' else 2

    condition_img = current_image.cuda()
    current_segm = current_segm.cuda()

    condition_meta = torch.tensor([[
        current_age / 100.0, current_dx, target_dx, current_mmse / 30.0, sex, target_age / 100.0
    ]]).cuda().float()

    alpha = torch.tensor([target_dx / 2.0]).cuda()

    model_args = {
        'condition_meta': condition_meta,
        'condition_img': condition_img,
        'alpha': alpha,
        'segm': current_segm,
        'M': None
    }

    input_flow = None
    mri, flow = model(input_flow, **model_args)

    mri = mri.cpu()[0][0].numpy()
    flows = flow.clone()[0].cpu().permute(1, 2, 3, 0).numpy()
    flow = flow.cpu()[0][0].numpy()

    if args.vis:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(18, 8))
        plt.subplot(3, 3, 1)
        plt.imshow(mri[mri.shape[0] // 2, :, :], cmap='gray')
        plt.subplot(3, 3, 2)
        plt.imshow(mri[:, mri.shape[1] // 2, :], cmap='gray')
        plt.subplot(3, 3, 3)
        plt.imshow(mri[:, :, mri.shape[2] // 2], cmap='gray')

        
        flow = np.linalg.norm(flows, axis=-1)
        plt.subplot(3, 3, 7)
        plt.imshow(flow[flow.shape[0] // 2, :, :], cmap='gray')
        plt.colorbar()
        plt.subplot(3, 3, 8)
        plt.imshow(flow[:, flow.shape[1] // 2, :], cmap='gray')
        plt.colorbar()
        plt.subplot(3, 3, 9)
        plt.imshow(flow[:, :, flow.shape[2] // 2], cmap='gray')
        plt.colorbar()
        plt.savefig(f'{args.output}/vis_{target_age}.png')
        plt.close()
        
    mri_nii = nib.Nifti1Image(mri, affine=const.MNI152_1P5MM_AFFINE)
    segm = nib.load(current_segm_path)
    segm = resample_from_to(segm, mri_nii, order=0)
    mask = segm.get_fdata() > 0
    mri[ mask == 0 ] = 0
    mri = np.clip(mri, 0.0, 1.0)

    pred_path = os.path.join(args.output, f'pred_{target_age}.nii.gz')
    mri_nii.to_filename(pred_path)


if __name__ == '__main__':
    main()