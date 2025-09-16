"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_msssim import ssim

def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def ncc_loss(I, J, device, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to(device)
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)


import torch
import torch.nn as nn

class GradientDifferenceLoss(nn.Module):
    """
    计算梯度差分损失 (L1或L2范数)。
    这是一种比Sobel更简单但同样有效的边缘约束方法。
    """
    def __init__(self, norm='L1'):
        super().__init__()
        assert norm in ['L1', 'L2']
        if norm == 'L1':
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()

    def _get_gradient(self, x):
        """
        计算 x 在三个维度上的梯度。
        x: (B, 1, D, H, W)
        """
        # 深度方向的梯度
        grad_d = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        # 高度方向的梯度
        grad_h = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        # 宽度方向的梯度
        grad_w = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return grad_d, grad_h, grad_w

    def forward(self, moved_image, fixed_image):
        """
        计算 moved_image 和 fixed_image 梯度之间的损失。
        """
        moved_grad_d, moved_grad_h, moved_grad_w = self._get_gradient(moved_image)
        fixed_grad_d, fixed_grad_h, fixed_grad_w = self._get_gradient(fixed_image)

        # 分别计算每个方向上的梯度损失，然后相加
        loss_d = self.loss(moved_grad_d, fixed_grad_d)
        loss_h = self.loss(moved_grad_h, fixed_grad_h)
        loss_w = self.loss(moved_grad_w, fixed_grad_w)
        
        return loss_d + loss_h + loss_w


def normal_alignment_loss(phi, normal_field):
    # norm = phi.norm(dim=1, keepdim=True)
    # index = norm > 0.01
    # index_expanded = index.expand(-1, 3, -1, -1, -1) # -1表示保持维度不变

    # phi_selected = phi[index_expanded].view(-1, 3) # 重新塑形为 (N, 3)
    # normal_selected = normal_field[index_expanded].view(-1, 3) # 重新塑形为 (N, 3)
    cosine_sim = F.cosine_similarity(phi, normal_field, dim=1) # dim=1 现在是正确的
    loss = 1.0 - cosine_sim.abs().mean()
    return loss


class VMX:
    def __init__(self):
        self.grad_diff = GradientDifferenceLoss()
    
    def training_losses(self, model, x_start, model_kwargs=None):
        # x start 为 fixed
        input_flow = None
        # segm = model_kwargs.pop('segm')
        # segm = (segm > 0).float()
        current_normal = model_kwargs.pop('current_normal')
        pred_img, flow, alpha_pred = model(input_flow, **model_kwargs)
        sim_loss = ncc_loss(pred_img, x_start, device='cuda')
        diff_loss = self.grad_diff(pred_img, x_start)
        # pred_img = segm * pred_img
        # x_start = segm * x_start
        # flow = segm * flow
        # sim_loss = F.l1_loss(pred_img, x_start) + 0.1 * (1 - ssim(pred_img, x_start, data_range=1.0, size_average=True))
        # grad_loss = gradient_loss(flow)
        grad_loss = normal_alignment_loss(current_normal, flow)

        alpha_gt = model_kwargs['alpha']
        cn_index = alpha_gt == 0
        ad_index = alpha_gt == 1
        alpha_cn = alpha_gt[cn_index]
        alpha_ad = alpha_gt[ad_index]

        alpha_loss = 0.0 * F.mse_loss(alpha_pred, alpha_gt.float())
        if alpha_cn.shape[0] > 0:
            alpha_loss += F.mse_loss(alpha_pred[cn_index], torch.zeros_like(alpha_pred[cn_index]))
        if alpha_ad.shape[0] > 0:
            alpha_loss += F.mse_loss(alpha_pred[ad_index], torch.ones_like(alpha_pred[ad_index]))
        

        total_loss = sim_loss + 1.0 * grad_loss + 0.00 * diff_loss + 0.1 * alpha_loss
        term = {}
        term['loss'] = total_loss
        term['sim_loss'] = sim_loss
        term['diff_loss'] = grad_loss
        term['alpha_loss'] = alpha_loss

        # term['grad_loss'] = grad_loss
        return term
