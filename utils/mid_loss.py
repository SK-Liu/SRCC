import torch
import torch.nn as nn
import torch.nn.functional as F

class MID_LOSS(nn.Module):
    def __init__(self, beta=0.3, wordvec_array=None, args=None):
        super(MID_LOSS, self).__init__()
        self.eps = 1e-7
        self.wordvec_array = wordvec_array
        self.embed_len = args.embed_len # embed_len -> 1024
        self.beta = beta
        self.num_fea = args.num_fea # num_fea -> 3

    """ def forward(self, x, y): # x-> output, y -> label
        batch_size = y.shape[0]
        loss = torch.zeros(batch_size).cuda()
        
        un_flat = x.view(x.shape[0], self.embed_len, -1) 
        M = un_flat.shape[2]

        dot_prod_all = [
            torch.sum(
                (un_flat[:, :, i].unsqueeze(2) * self.wordvec_array), dim=1
            ).unsqueeze(2)
            for i in range(M)
        ]

        dot_prod_all = torch.max(torch.cat(dot_prod_all, dim=2), dim=-1)
        dot_prod_all = dot_prod_all.values """
    def forward(self, x, y):  # x-> output, y -> label
        batch_size = y.shape[0]
        loss = torch.zeros(batch_size, device=x.device)

        un_flat = x.view(x.shape[0], self.embed_len, -1)
        M = un_flat.shape[2]

        # [C, num_classes]，沿着语义维做归一化
        wordvec = F.normalize(self.wordvec_array, p=2, dim=0)

        dot_prod_all = [
            torch.sum(
                F.normalize(un_flat[:, :, i], p=2, dim=1).unsqueeze(2) * wordvec,
                dim=1
            ).unsqueeze(2)
            for i in range(M)
        ]

        dot_prod_all = torch.max(torch.cat(dot_prod_all, dim=2), dim=-1).values

        for i in range(0, batch_size):
            dot_prod_pos = dot_prod_all[i, y[i] == 1]  
            dot_prod_neg = dot_prod_all[
                i, (1 - y[i]).bool()
            ]  
            if len(dot_prod_neg) == 0:  
                v = -dot_prod_pos.unsqueeze(1)
            else:
                v = dot_prod_neg.unsqueeze(0) - dot_prod_pos.unsqueeze(1)

            num_pos = dot_prod_pos.shape[0]
            total_var = calc_diversity(self.wordvec_array, y[i])
            if self.num_fea == 1:
                loss[i] = torch.sum(torch.log(1 + torch.exp(v))) / (num_pos)
            else:
                loss[i] = (
                    (1 + total_var) * torch.sum(torch.log(1 + torch.exp(v))) / (num_pos)
                )
                l1_err = var_regularization(un_flat[i])
                loss[i] = 2 * ((1 - self.beta) * (loss[i]) + self.beta * l1_err)

        return loss.mean()
""" class MID_LOSS(nn.Module):
    def __init__(self, beta=0.3, wordvec_array=None, args=None):
        super(MID_LOSS, self).__init__()
        self.eps = 1e-7
        self.beta = beta

        # 词向量：期望形状 [embed_len, num_classes]
        w = torch.as_tensor(wordvec_array, dtype=torch.float32)
        # 有的实现会是 [1, C, D] 或 [C, D]，这里统一成 [D, C]
        if w.dim() == 3:
            w = w.squeeze(0)
        if w.shape[0] != args.embed_len and w.shape[1] == args.embed_len:
            # 形状是 [C, D]，转成 [D, C]
            w = w.transpose(0, 1)

        self.embed_len = w.shape[0]          # 1024
        self.num_classes = w.shape[1]
        self.num_fea = args.num_fea          # 这里记得设成和 v 一致，比如 7

        # 用 buffer 确保自动跟随模型到 cuda
        self.register_buffer("wordvec_array", w)

    def forward(self, x, y):   # x: [B, 1024, M], y: [B, C]
        # 统一 dtype / device
        x = x.to(self.wordvec_array.device, dtype=torch.float32)
        y = y.to(self.wordvec_array.device, dtype=torch.float32)

        batch_size = y.shape[0]
        loss = x.new_zeros(batch_size)

        # 安全检查
        if x.numel() % self.embed_len != 0:
            raise RuntimeError(
                f"MID_LOSS: x.numel()={x.numel()} 不能被 embed_len={self.embed_len} 整除"
            )

        # x: [B, D, M]
        un_flat = x.view(x.shape[0], self.embed_len, -1)   # [B, D, M]
        M = un_flat.shape[2]

        # 向量化计算：所有 descriptor 与所有 label 语义向量的点积
        # un_flat: [B, D, M] -> [B*M, D]
        pd = un_flat.permute(0, 2, 1).reshape(-1, self.embed_len)   # [B*M, D]
        # wordvec_array: [D, C]
        dp = pd @ self.wordvec_array                               # [B*M, C]
        # reshape 回来： [B, M, C] -> [B, C, M]
        dot_all = dp.view(x.shape[0], M, self.num_classes).permute(0, 2, 1)
        # 每个类取所有 feature 里最大的得分: [B, C]
        dot_all, _ = dot_all.max(dim=2)

        for i in range(batch_size):
            pos_mask = (y[i] == 1)
            neg_mask = ~pos_mask

            dot_pos = dot_all[i, pos_mask]      # [P]
            dot_neg = dot_all[i, neg_mask]      # [N]

            num_pos = dot_pos.numel()
            num_neg = dot_neg.numel()

            if num_pos == 0 or num_neg == 0:
                # 没有正类或没有负类，跳过这个样本，loss=0
                print(f"[WARN] MID_LOSS: batch {i} 中没有正类或负类，跳过该样本")
                continue

            # pairwise 差：[N, P]
            v = dot_neg.unsqueeze(1) - dot_pos.unsqueeze(0)
            # 数值安全：防止 log(1+exp(v)) 爆
            v = torch.clamp(v, min=-10.0, max=10.0)

            # 语义多样性
            total_var = calc_diversity(self.wordvec_array, y[i])

            base = torch.log1p(torch.exp(v)).sum() / max(float(num_pos), 1.0)

            if self.num_fea == 1:
                loss_i = base
            else:
                l1_err = var_regularization(un_flat[i])  # un_flat[i]: [D, M]
                loss_i = (1 + total_var) * base
                loss_i = 2.0 * ((1 - self.beta) * loss_i + self.beta * l1_err)

            loss[i] = loss_i

        return loss.mean()
 """

def calc_diversity(wordvec_array, y_i):
    rel_vecs = wordvec_array[:, :, y_i == 1]
    rel_vecs = rel_vecs.squeeze(0)
    if rel_vecs.shape[1] == 1:
        sig = rel_vecs * 0
    else:
        sig = torch.var(rel_vecs, dim=1)

    return sig.sum()

""" def calc_diversity(wordvec_array, y_i):
    
    #wordvec_array: [D, C]
    #y_i: [C]，0/1，表示该样本的正类
    #返回：正类语义向量在 embedding 维度上的方差之和
    
    # 保证在同一 device & 1D
    y_i = y_i.to(wordvec_array.device).view(-1)     # [C]
    pos_mask = (y_i == 1)                           # bool [C]

    # 没有正类，直接返回 0
    if pos_mask.sum() <= 1:
        return wordvec_array.new_tensor(0.0)

    # 取出所有正类对应的语义向量: [D, num_pos]
    rel_vecs = wordvec_array[:, pos_mask]           # [D, P]

    # 在“类别维度”上算方差：得到每个 embedding 维度的方差 [D]
    sig = torch.var(rel_vecs, dim=1, unbiased=False)

    # 再把所有维度的方差加起来，得到一个标量
    return sig.sum() """



def var_regularization(x_i):
    sig2 = torch.var(x_i, dim=1)
    l1_err = torch.norm(sig2, dim=-1, p=1)
    return l1_err
import torch
import torch.nn as nn


import torch
import torch.nn as nn


class AsymmetricLossClassWiseSmooth(nn.Module):
    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        eps=1e-8,
        class_smooth=None,
        reduction='mean'
    ):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

        if class_smooth is None:
            raise ValueError("class_smooth must be provided")

        self.register_buffer(
            "class_smooth",
            torch.tensor(class_smooth, dtype=torch.float32)
        )

    def forward(self, logits, targets):

        targets = targets.float()

        # ---------- class-wise label smoothing ----------
        smooth = self.class_smooth.unsqueeze(0)

        targets = targets * (1 - smooth) + (1 - targets) * smooth

        # ---------- sigmoid ----------
        prob = torch.sigmoid(logits)

        prob_pos = prob
        prob_neg = 1 - prob

        # ---------- asymmetric clipping ----------
        if self.clip is not None and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1)

        # ---------- log loss ----------
        los_pos = targets * torch.log(prob_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(prob_neg.clamp(min=self.eps))

        loss = los_pos + los_neg

        # ---------- asymmetric focusing ----------
        if self.gamma_neg > 0 or self.gamma_pos > 0:

            pt = prob_pos * targets + prob_neg * (1 - targets)

            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)

            one_sided_weight = torch.pow(1 - pt, gamma)

            loss *= one_sided_weight

        loss = -loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss