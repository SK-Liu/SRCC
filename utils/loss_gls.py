
from sympy import gamma


def binarize_vec(v):
    tmp = v.clone()
    tmp[tmp != 0.0] = 1.0
    #vartheta = torch.tensor([freq_c / sum(freq) for c in range(C)])  # freq_c 为训练集正样本比例
    #gamma_vec = gamma * vartheta
    return tmp

import torch

def _check_prob(name, x):
    if not torch.isfinite(x).all():
        idx = (~torch.isfinite(x)).nonzero(as_tuple=False)[0]
        raise RuntimeError(f"{name} NaN/Inf at {idx.tolist()} val={x[tuple(idx)].item()}")
    mn, mx = x.min().item(), x.max().item()
    if mn < 0.0 or mx > 1.0:
        idx = ((x < 0) | (x > 1)).nonzero(as_tuple=False)[0]
        raise RuntimeError(f"{name} out of [0,1]: min={mn} max={mx}, idx={idx.tolist()}, val={x[tuple(idx)].item()}")
def loss_gls(p, y_tilde, y_knn, smooth_rate, beta_l, criterion, hist=0, L=2):
    """ gamma = beta_l / 2
    y_bar = (
        (1 - smooth_rate) * y_tilde + smooth_rate * ( gamma * 1 + (1 - gamma) * y_knn)
    ) * binarize_vec(y_knn + y_tilde)
    _check_prob("p(input)", p)
    _check_prob("y_bar(target)", y_bar)
    loss = criterion(p, y_bar)
    return loss """
    p = p.float()
    p = p.clamp(1e-6, 1 - 1e-6)
    y_tilde = y_tilde.float().clamp(0.0, 1.0)
    y_knn = y_knn.float().clamp(0.0, 1.0)
    smooth_rate = smooth_rate.float().clamp(0.0, 1.0)

    gamma = beta_l / 2.0
    #gamma = max(0.0, min(1.0, gamma))  # 先硬限制到[0,1]

    aux_target = gamma + (1.0 - gamma) * y_knn
    aux_target = aux_target.clamp(0.0, 1.0)

    mask = binarize_vec(y_knn + y_tilde).float()

    y_bar = ((1.0 - smooth_rate) * y_tilde + smooth_rate * aux_target) * mask
    y_bar = y_bar.clamp(0.0, 1.0)

    _check_prob("p(input)", p)
    _check_prob("y_bar(target)", y_bar)

    loss = criterion(p, y_bar)
    return loss

import torch.nn.functional as F

def loss_gls_logits(logits, y_tilde, y_knn, smooth_rate, beta_l):
    """ gamma = max(0.0, min(1.0, beta_l / 2))
    mask = binarize_vec(y_knn + y_tilde).float()
    y_bar = ((1 - smooth_rate) * y_tilde +
             smooth_rate * (gamma * 1.0 + (1 - gamma) * y_knn)) * mask
    y_bar = y_bar.clamp(0.0, 1.0) """
    gamma = max(0.0, min(1.0, beta_l / 2))
    pos_mask = (y_tilde > 0.5).float()
    neg_mask = 1.0 - pos_mask

    y_bar_pos = (1 - smooth_rate) * y_tilde + smooth_rate * (gamma + (1 - gamma) * y_knn)
    y_bar = pos_mask * y_bar_pos + neg_mask * y_tilde
    y_bar = y_bar.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(logits, y_bar)


import torch
import torch.nn.functional as F

def focal_loss_tail(pred_prob, labels, tail_mask, gamma=2.5, eps=1e-7):
    """
    pred_prob: [B, C], sigmoid 后的概率
    labels:    [B, C]
    tail_mask: [C] bool tensor（True 表示 tail 类）
    """
    pred_prob = torch.clamp(pred_prob, eps, 1.0 - eps)

    BCE = -(labels * torch.log(pred_prob) +
            (1 - labels) * torch.log(1 - pred_prob))   # [B, C]

    pt = labels * pred_prob + (1 - labels) * (1 - pred_prob)
    mod = (1 - pt) ** gamma

    FL = mod * BCE                   # [B, C]
    FL_tail = FL[:, tail_mask]       # 只对 tail 类取平均

    if FL_tail.numel() == 0:
        return torch.tensor(0.0, device=pred_prob.device)
    return FL_tail.mean()


def loss_gls_head_tail(p, y_tilde, y_knn, head_mask, args, eps=1e-7):
    """
    p:       [B, C] 概率（sigmoid 后）
    y_tilde: [B, C] 当前使用的标签（建议传 pseudo_labels[idx]）
    y_knn:   [B, C] NSD 邻居投票（计数 / topk 归一化）
    head_mask: [C] bool，True 表示 head 类
    """
    device = p.device
    B, C = p.shape

    # 全局参数
    gamma = args.beta_l / 2
    smooth_rate = args.smooth_rate   # 原来的 smooth_rate

    # head 类用完整的 GLS，tail 类不做 GLS（只保留 y_tilde）
    lam = torch.full((C,), smooth_rate, device=device)
    lam[~head_mask.to(device)] = 0.0   # tail 类 smooth_rate = 0，相当于 y_bar = y_tilde

    # [B, C]
    lam = lam.unsqueeze(0).expand(B, -1)

    # 这个相当于论文里的 gamma * 1 + (1 - gamma) * y_knn
    soft_part = gamma * 1.0 + (1.0 - gamma) * y_knn

    # binary mask: 只在 y_tilde 或 y_knn 激活的位置监督
    mask = ((y_tilde + y_knn) > 0).float()

    y_bar = ((1.0 - lam) * y_tilde + lam * soft_part) * mask

    # 用 BCE 计算 head 部分的损失
    p = torch.clamp(p, eps, 1.0 - eps)
    bce = -(y_bar * torch.log(p) + (1.0 - y_bar) * torch.log(1.0 - p))  # [B, C]
    bce_head = bce[:, head_mask.to(device)]

    if bce_head.numel() == 0:
        return torch.tensor(0.0, device=device)
    return bce_head.mean()


def hybrid_loss(
    pred_logits,
    labels_noisy,          # 原 noisy labels [B, C]
    pseudo_labels_batch,   # 当前使用的伪标签 [B, C]
    knn_labels_batch,      # NSD neighbor labels [B, C]（或计数）
    v,                     # MID 用的 descriptor
    mid_criteria,
    head_mask,
    tail_mask,
    epoch,
    args,
):
    device = pred_logits.device
    pred_prob = torch.sigmoid(pred_logits)

    # === 分类主干 BCE，用当前伪标签 ===
    bce_main = F.binary_cross_entropy(
        pred_prob,
        pseudo_labels_batch
    )

    # === head 上的 GLS（用 pseudo_labels + y_knn） ===
    # y_knn 需要归一化成 [0,1]，假设传进来前已经 / nsd_topk
    loss_gls_head = loss_gls_head_tail(
        pred_prob,
        y_tilde=pseudo_labels_batch,
        y_knn=knn_labels_batch,
        head_mask=head_mask,
        args=args
    )

    # === tail 上的 focal loss（用 pseudo_labels 监督） ===
    loss_focal_tail = focal_loss_tail(
        pred_prob,
        pseudo_labels_batch,
        tail_mask=tail_mask.to(device),
        gamma=2.0
    )

    # === MID loss（用 union 后的标签比较稳，你可以先用 pseudo_labels_batch） ===
    loss_mid = mid_criteria(v, pseudo_labels_batch)

    # === 不同阶段给不同权重（这里只是一个合理的初始设置）===
    if epoch < 10:
        lambda_bce   = 1.0
        lambda_gls   = 0.0
        lambda_focal = 0.0
        lambda_mid   = 0.0
    elif epoch < 15:
        lambda_bce   = 0.6
        lambda_gls   = 0.2
        lambda_focal = 0.19
        lambda_mid   = 0.01
    else:
        lambda_bce   = 0.4
        lambda_gls   = 0.2
        lambda_focal = 0.29
        lambda_mid   = 0.01

    loss = (lambda_bce   * bce_main +
            lambda_gls   * loss_gls_head +
            lambda_focal * loss_focal_tail +
            lambda_mid   * loss_mid)

    return loss, {
        "bce": bce_main.item(),
        "gls": loss_gls_head.item(),
        "focal": loss_focal_tail.item(),
        "mid": loss_mid.item(),
    }
