import os.path as osp
import torch
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
from monai import transforms
import pickle
import numpy as np
from collections import OrderedDict
from utils import const
import torchio as tio
import torch.nn.functional as F
import kornia


def get_advanced_transforms():
    full_transform = tio.Compose([
        tio.RandomFlip(axes=('LR',), p=0.5),
        tio.RandomAffine(
            scales=(0.85, 1.15),
            degrees=15,
            translation=12,
            isotropic=False, 
            default_pad_value='minimum',
        ),
    ])
    return full_transform


def generate_soft_affiliation_matrix(
    segm_map,
    num_regions,
    patch_size=(16, 16, 16),
):
    pD, pH, pW = patch_size
    B, D, H, W = segm_map.shape
    device = segm_map.device

    gD = (D + pD - 1) // pD
    gH = (H + pH - 1) // pH
    gW = (W + pW - 1) // pW
    num_patches = gD * gH * gW

    seg_one_hot = F.one_hot(segm_map.long(), num_classes=num_regions).float()

    seg_one_hot = seg_one_hot.permute(0, 4, 1, 2, 3)

    affiliation_grid = F.avg_pool3d(
        seg_one_hot,
        kernel_size=patch_size,
        stride=patch_size,
        padding=0 # 无重叠
    )

    M = affiliation_grid.reshape(B, num_regions, num_patches)
    M = M.transpose(1, 2)

    total_valid_affiliation = M.sum(dim=2, keepdim=True)

    M_normalized = M / total_valid_affiliation.clamp(min=1e-6)

    return M_normalized


def segs(seg):
    # seg: BDHW
    index = 0
    for code in seg.unique().tolist():
        seg[seg == code] = index
        index += 1

    seg = F.pad(seg, (0, 8, 0, 0, 0, 8), mode='constant', value=0)

    M = generate_soft_affiliation_matrix(
        seg,
        33,
        patch_size=(16, 16, 16),
    )

    return M


def get_gaussian_kernel_3d(kernel_size: int, sigma: float) -> torch.Tensor:
    kernel_1d = torch.arange(kernel_size).float()
    kernel_1d -= kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()

    kernel_3d = torch.einsum('i,j,k->ijk', kernel_1d, kernel_1d, kernel_1d)
    
    kernel_3d /= kernel_3d.sum()
    return kernel_3d


def gaussian_blur_3d(
    image_tensor: torch.Tensor, 
    kernel_size: int, 
    sigma: float
) -> torch.Tensor:
    B, C, D, H, W = image_tensor.shape
    kernel = get_gaussian_kernel_3d(kernel_size, sigma).to(image_tensor.device, dtype=image_tensor.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(C, 1, 1, 1, 1)
    padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2)
    padded_image = F.pad(image_tensor, padding, mode='replicate')
    blurred_image = F.conv3d(padded_image, kernel, padding='valid', groups=C)
    return blurred_image


def _ensure_5d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3: return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 4: return tensor.unsqueeze(0)
    return tensor

def get_normal_field(
    image_tensor: torch.Tensor, 
    sigma: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    if device is None:
        device = image_tensor.device
        
    image_tensor_5d = _ensure_5d(image_tensor).to(device)

    kernel_size = int(2 * sigma + 1)
    if kernel_size % 2 == 0: kernel_size += 1
    
    smoothed_image = gaussian_blur_3d(image_tensor_5d, kernel_size, sigma)
    
    grads = kornia.filters.spatial_gradient3d(smoothed_image, order=1).squeeze(1)
    magnitude = torch.linalg.norm(grads, dim=1, keepdim=True) + 1e-9
    normal_field = grads / magnitude
    
    return normal_field



class ADNIMRIPairDataset(Dataset):
    def __init__(self, data_path, datalist, mpath=None, stage2=False):
        super().__init__()
        self.data = pd.read_csv(datalist)
        if not stage2:
            self.data = self.data[self.data['image2_dx'] == 'CN']
        else:
            self.data = self.data[self.data['image2_dx'] != 'CN']

        self.data_load = torch.load(data_path, weights_only=False)
        self.segm_load = torch.load(data_path.replace('.pt', '_segm.pt'), weights_only=False)
        self.mpath = mpath
        self.trans = get_advanced_transforms()
        
    def __len__(self):
        return len(self.data)
    
    def parse_dx(self, str):
        if str == 'CN':
            return 0
        elif str == 'MCI':
            return 1
        elif str == 'Dementia':
            return 2
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        img1 = self.data_load[data['image1_name']]
        img2 = self.data_load[data['image2_name']]
        segm1 = self.segm_load[data['image1_name']]
        segm2 = self.segm_load[data['image2_name']]
        # 数据增强
        if False:
            # 2. 创建一个 Torchio Subject 对象，这是同步变换的关键
            subject = tio.Subject(
                img1=tio.ScalarImage(tensor=img1),
                img2=tio.ScalarImage(tensor=img2),
                segm1=tio.LabelMap(tensor=segm1),
                segm2=tio.LabelMap(tensor=segm2)
            )
            
            # 3. 应用完整的变换流程
            transformed_subject = self.trans(subject)
            
            # 4. 从变换后的subject中取回数据
            img1 = transformed_subject.img1.data
            img2 = transformed_subject.img2.data
            segm1 = transformed_subject.segm1.data
            segm2 = transformed_subject.segm2.data

        age1 = data['image1_age'].clip(0, 100) / 100.0
        age2 = data['image2_age'].clip(0, 100) / 100.0
        sex = 1 if data['sex'] == 'Female' else 0
        dx1 = self.parse_dx(data['image1_dx'])
        dx2 = self.parse_dx(data['image2_dx'])
        mmse1 = data['image1_mmse']
        mmse2 = data['image2_mmse']
        # group = data['group']
        group = data['image2_dx']
        m1, m2 = None, None
        if self.mpath:
            m1 = np.load(osp.join(self.mpath, f"{data['image1_name']}.npy"))
            m2 = np.load(osp.join(self.mpath, f"{data['image2_name']}.npy"))
        return {
            'image1': img1,
            'image2': img2,
            'segm1': segm1,
            'segm2': segm2,
            'age1': age1,
            'age2': age2,
            'dx1': dx1,
            'dx2': dx2,
            'mmse1': mmse1,
            'mmse2': mmse2,
            'sex': sex,
            'image1_name': data['image1_name'],
            'image2_name': data['image2_name'],
            'group': group,
            'm1': m1,
            'm2': m2
        }


class ADNILatentPairDataset(Dataset):
    def __init__(self, data_path, datalist, mpath=None):
        super().__init__()
        self.data = pd.read_csv(datalist)
        # self.data = self.data[self.data['group'] != 'CN']
        self.data_path = data_path
        self.mpath = mpath
    
    def __len__(self):
        return len(self.data)
    
    def parse_dx(self, str):
        if str == 'CN':
            return 0
        elif str == 'MCI':
            return 1
        elif str == 'Dementia':
            return 2
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        img1 = np.load(osp.join(self.data_path, f"{data['image1_name']}.npy"))
        img1 = torch.from_numpy(img1)
        img2 = np.load(osp.join(self.data_path, f"{data['image2_name']}.npy"))
        img2 = torch.from_numpy(img2)
        age1 = data['image1_age'].clip(0, 100) / 100.0
        age2 = data['image2_age'].clip(0, 100) / 100.0
        sex = 1 if data['sex'] == 'Female' else 0
        dx1 = self.parse_dx(data['image1_dx'])
        dx2 = self.parse_dx(data['image2_dx'])
        mmse1 = data['image1_mmse']
        mmse2 = data['image2_mmse']
        # group = data['group']
        group = data['image2_dx']
        m1, m2 = None, None
        if self.mpath:
            m1 = np.load(osp.join(self.mpath, f"{data['image1_name']}.npy"))
            m2 = np.load(osp.join(self.mpath, f"{data['image2_name']}.npy"))
        return {
            'image1': img1,
            'image2': img2,
            'age1': age1,
            'age2': age2,
            'dx1': dx1,
            'dx2': dx2,
            'mmse1': mmse1,
            'mmse2': mmse2,
            'sex': sex,
            'image1_name': data['image1_name'],
            'image2_name': data['image2_name'],
            'group': group,
            'm1': m1,
            'm2': m2
        }